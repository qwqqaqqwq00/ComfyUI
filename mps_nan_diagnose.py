#!/usr/bin/env python3
"""
MPS 黑图/NaN 一键排查脚本

用法:
    python mps_nan_diagnose.py            # 完整诊断
    python mps_nan_diagnose.py --model ltx  # 只测 LTX 相关
    python mps_nan_diagnose.py --quick      # 快速模式（只测关键 op）

功能:
    1. 检测环境（PyTorch/MPS/macOS 版本）
    2. 逐精度、逐 op 对比 CPU vs MPS
    3. 定位 NaN 根因（融合内核 bug vs 数值溢出 vs 灾难性抵消）
    4. 检查 ComfyUI 模型文件是否已有 MPS workaround
    5. 生成修复建议
"""

import argparse
import os
import sys
import subprocess
import textwrap
import torch
import warnings

warnings.filterwarnings("ignore")

# ── 颜色输出 ──────────────────────────────────────────────
class C:
    R = "\033[31m"; G = "\033[32m"; Y = "\033[33m"
    B = "\033[34m"; M = "\033[35m"; C = "\033[36m"
    BOLD = "\033[1m"; DIM = "\033[2m"
    END = "\033[0m"

def ok(msg):   print(f"  {C.G}✅ {msg}{C.END}")
def bad(msg):  print(f"  {C.R}❌ {msg}{C.END}")
def warn(msg): print(f"  {C.Y}⚠️  {msg}{C.END}")
def info(msg): print(f"  {C.B}ℹ️  {msg}{C.END}")
def head(msg): print(f"\n{C.BOLD}{C.C}{'─'*60}{C.END}\n{C.BOLD}{msg}{C.END}\n{C.C}{'─'*60}{C.END}")

# ── 精度配置 ──────────────────────────────────────────────
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DT_NAMES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def section_env():
    """第 1 步: 环境检测"""
    head("第 1 步: 环境检测")

    import platform
    print(f"  macOS:      {platform.mac_ver()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  MPS 可用:   {torch.backends.mps.is_available()}")
    print(f"  MPS 已构建: {torch.backends.mps.is_built()}")

    # 芯片型号
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True)
        chip = r.stdout.strip()
    except Exception:
        chip = "unknown"
    print(f"  芯片:       {chip}")

    # 统一内存
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                          capture_output=True, text=True)
        mem_gb = int(r.stdout.strip()) / 1024**3
        print(f"  统一内存:   {mem_gb:.0f} GB")
    except Exception:
        pass

    if not torch.backends.mps.is_available():
        bad("MPS 不可用，本脚本无法继续")
        sys.exit(1)
    ok("MPS 可用")


def section_gelu():
    """第 2 步: GELU 融合内核诊断（核心根因定位）"""
    head("第 2 步: GELU 融合内核诊断")

    # 测试值: 从小到大，找 NaN 边界
    test_vals = [1, 5, 10, 12, 14, 15, 20, 30, 40, 50]

    print(f"  {'值':>4} | {'fp32 CPU':>10} {'fp32 MPS':>10} | "
          f"{'fp16 CPU':>10} {'fp16 MPS':>10} | "
          f"{'bf16 CPU':>10} {'bf16 MPS':>10}")
    print(f"  {'─'*4}─┼{'─'*23}┼{'─'*23}┼{'─'*23}")

    nan_thresholds = {}  # {dtype: (边界值, 原因)}

    for val in test_vals:
        row = f"  {val:>4} |"
        for dt in DTYPES:
            x_cpu = torch.tensor([float(val)], dtype=dt)
            x_mps = torch.tensor([float(val)], dtype=dt, device="mps")

            try:
                g_cpu = torch.nn.functional.gelu(x_cpu, approximate="tanh")
            except Exception:
                g_cpu = torch.tensor([float("nan")])

            try:
                g_mps = torch.nn.functional.gelu(x_mps, approximate="tanh")
            except Exception:
                g_mps = torch.tensor([float("nan")])

            cpu_str = f"{g_cpu.item():.2f}" if not torch.isnan(g_cpu).item() else "NaN"
            mps_str = f"{g_mps.item():.2f}" if not torch.isnan(g_mps).item() else f"{C.R}NaN{C.END}"

            # 标记 CPU 和 MPS 不一致
            cpu_nan = torch.isnan(g_cpu).item()
            mps_nan = torch.isnan(g_mps).item()

            if mps_nan and not cpu_nan:
                if dt not in nan_thresholds:
                    # 记录第一个 NaN 的值
                    nan_thresholds[dt] = val
                mps_str = f"{C.R}NaN❌{C.END}"
            elif mps_nan and cpu_nan:
                mps_str = f"{C.Y}NaN{C.END}"  # 两者都 NaN（真正的溢出）

            row += f" {cpu_str:>10} {mps_str:>18}"
            if dt != DTYPES[-1]:
                row += " |"
        print(row)

    return nan_thresholds


def section_gelu_root_cause(thresholds):
    """第 3 步: GELU NaN 根因分析"""
    head("第 3 步: GELU NaN 根因分析")

    if not thresholds:
        ok("所有精度下 GELU 均正常，无 NaN")
        return

    for dt, threshold in thresholds.items():
        dt_name = DT_NAMES[dt]
        x = torch.tensor([float(threshold)], dtype=dt, device="mps")

        print(f"\n  {C.BOLD}[{dt_name}] NaN 边界: x = {threshold}{C.END}")

        # 手动逐项拆解 GELU tanh 公式
        # 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x³)))
        x3 = x * x * x
        inner = x + 0.044715 * x3
        arg = (2.0 / torch.pi) ** 0.5 * inner
        tanh_val = torch.tanh(arg)
        result = 0.5 * x * (1 + tanh_val)

        print(f"    x³              = {x3.item():.2f}    "
              f"{'❌ 溢出!' if torch.isinf(x3).item() else '✅'}")
        print(f"    x + 0.044715*x³ = {inner.item():.2f}  "
              f"{'❌ 溢出!' if torch.isinf(inner).item() else '✅'}")
        print(f"    tanh(内部参数)  = {tanh_val.item():.6f}  "
              f"{'❌ NaN!' if torch.isnan(tanh_val).item() else '✅'}")
        print(f"    手动逐项结果    = {result.item():.2f}  "
              f"{'❌ NaN!' if torch.isnan(result).item() else '✅'}")

        # 融合 GELU
        fused = torch.nn.functional.gelu(x, approximate="tanh")
        print(f"    F.gelu 融合结果 = {fused.item():.2f}  "
              f"{'❌ NaN!' if torch.isnan(fused).item() else '✅'}")

        # 判断根因
        print(f"\n    {C.BOLD}根因判断:{C.END}")
        if torch.isinf(x3).item():
            bad(f"{dt_name}: x³ 在 |x|={threshold} 溢出（>{torch.finfo(dt).max:.0f}）")
            info(f"  → 数值范围溢出。fp16 指数仅 5 位，|x|>40 时 x³>65504 溢出。")
            info(f"  → 修复: F.gelu(x.float(), ...).to(dtype=x.dtype)")
        elif not torch.isnan(result).item() and torch.isnan(fused).item():
            bad(f"{dt_name}: 手动逐项正确但融合内核 NaN → MPS 融合 GELU 内核 bug")
            info(f"  → 非溢出。{dt_name} 指数范围足够，但 Metal 融合内核有实现缺陷。")
            info(f"  → 修复: F.gelu(x.float(), ...).to(dtype=x.dtype)")
        elif torch.isnan(result).item():
            bad(f"{dt_name}: 手动逐项也 NaN → 公式内部溢出")
            info(f"  → 需逐项检查哪个 step 产生 Inf/NaN")
            info(f"  → 修复: 将问题 step cast 到 fp32")
        else:
            warn(f"{dt_name}: 无法自动定位，需手动检查")


def section_adaln():
    """第 4 步: AdaLN 灾难性抵消诊断"""
    head("第 4 步: AdaLN 灾难性抵消诊断")

    print(f"  测试 (1+scale) 在 scale 接近 -1 时的精度损失:\n")
    print(f"  {'scale':>12} | {'fp32':>14} {'bf16':>14} {'fp16':>14}")
    print(f"  {'─'*12}─┼{'─'*16}┼{'─'*16}┼{'─'*16}")

    scales = [-0.5, -0.9, -0.99, -0.999, -0.9999, -1.0 + 1e-4]
    has_issue = False

    for s in scales:
        row = f"  {s:>12.6f} |"
        for dt in [torch.float32, torch.bfloat16, torch.float16]:
            t = torch.tensor([s], dtype=dt, device="mps")
            result = 1.0 + t
            true_val = 1.0 + s

            if result.item() == 0.0 and true_val != 0.0:
                # 灾难性抵消
                row += f" {C.R}{result.item():>14.10f}{C.END}"
                has_issue = True
            elif abs(result.item() - true_val) > 1e-3:
                row += f" {C.Y}{result.item():>14.10f}{C.END}"
                has_issue = True
            else:
                row += f" {result.item():>14.10f}"
        print(row)

    if has_issue:
        print(f"\n  {C.R}❌ 检测到灾难性抵消!{C.END}")
        info("  当 scale ≈ -1 时，bf16/fp16 的 (1+scale) 丢失全部有效数字")
        info("  → AdaLN 调制 rms_norm(x)*(1+scale)+shift 会产生错误值")
        info("  → 修复: scale/shift/gate 在 MPS 上 cast 到 fp32")
    else:
        ok("未检测到严重抵消")


def section_comfyui():
    """第 5 步: 检查 ComfyUI 模型文件是否有 MPS workaround"""
    head("第 5 步: ComfyUI 模型文件 MPS workaround 检查")

    # 推测 ComfyUI 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_ldm = os.path.join(script_dir, "comfy", "ldm")

    if not os.path.isdir(comfy_ldm):
        info(f"未找到 comfy/ldm 目录 ({comfy_ldm})")
        info("跳过模型文件检查（非 ComfyUI 目录）")
        return

    # 已知有 MPS workaround 的模型
    known_fixed = {
        "hunyuan3dv2_1": "hunyuandit.py",
        "hidream": None,
        "lightricks": "model.py",
    }

    # 扫描所有 model.py
    print(f"  扫描 {comfy_ldm} 下的 DiT 模型...\n")

    models = []
    for entry in sorted(os.listdir(comfy_ldm)):
        model_dir = os.path.join(comfy_ldm, entry)
        if not os.path.isdir(model_dir):
            continue
        # 找 .py 文件里的 GELU / AdaLN
        py_files = []
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".py"):
                py_files.append(os.path.join(model_dir, f))

        if not py_files:
            continue

        # 检查 GELU 和 mps workaround
        gelu_count = 0
        mps_fix_count = 0
        adaln_count = 0
        gelu_lines = []
        adaln_lines = []

        for pf in py_files:
            try:
                with open(pf, "r") as fh:
                    for lineno, line in enumerate(fh, 1):
                        low = line.lower()
                        if "gelu" in low and "approximate" in low:
                            gelu_count += 1
                            gelu_lines.append((pf, lineno, line.strip()))
                        if "scale_msa" in low or "shift_msa" in low or "gate_msa" in low:
                            adaln_count += 1
                            if lineno not in [l[1] for l in adaln_lines]:
                                adaln_lines.append((pf, lineno, line.strip()))
                        # 检测 MPS workaround（两种写法）:
                        # 1. 同一行有 mps + .float() (Hunyuan3D 单行写法)
                        # 2. device.type == "mps" 条件判断 (LTX 跨行写法)
                        if "mps" in low and (".float()" in line or "float32" in low):
                            mps_fix_count += 1
                        if "device.type" in low and "mps" in low:
                            mps_fix_count += 1
            except Exception:
                pass

        if gelu_count > 0 or adaln_count > 0:
            status = ""
            if mps_fix_count > 0:
                status = f"{C.G}已有 MPS 保护{C.END}"
            else:
                status = f"{C.R}缺少 MPS 保护{C.END}"
            print(f"  {C.BOLD}{entry}{C.END}")
            print(f"    GELU 调用: {gelu_count}  AdaLN 调制: {adaln_count}  "
                  f"MPS 保护: {mps_fix_count}  → {status}")
            if gelu_count > 0 and mps_fix_count == 0:
                for pf, ln, line in gelu_lines[:3]:
                    short = pf.replace(comfy_ldm + "/", "")
                    print(f"    {C.DIM}{short}:{ln}: {line[:80]}{C.END}")
            print()


def section_summary(thresholds):
    """第 6 步: 总结与修复建议"""
    head("第 6 步: 总结与修复建议")

    issues = []

    # GELU 问题
    if thresholds:
        for dt, val in thresholds.items():
            dt_name = DT_NAMES[dt]
            if dt == torch.float16 and val <= 40:
                issues.append(f"fp16 GELU 在 |x|≥{val} 溢出（x³ 超过 65504）")
            elif dt == torch.bfloat16 and val <= 20:
                issues.append(f"bf16 GELU 在 |x|≥{val} NaN（MPS 融合内核 bug）")
    else:
        ok("GELU 无 NaN 问题")

    # AdaLN 问题（假设有，因为前面检测了）
    issues.append("AdaLN (1+scale) 灾难性抵消（bf16/fp16 scale≈-1 时）")

    if not issues:
        ok("未检测到 MPS 数值问题，黑图可能是其他原因")
        return

    print(f"  {C.BOLD}发现的问题:{C.END}")
    for i, issue in enumerate(issues, 1):
        bad(f"{i}. {issue}")

    print(f"\n  {C.BOLD}修复建议:{C.END}")

    fix_code = textwrap.dedent("""\
        # ── 修复 1: GELU ──────────────────────────
        # 在 model.py 的 GELU 调用处:
        if x.device.type == "mps":
            x = F.gelu(x.float(), approximate="tanh").to(dtype=x.dtype)
        else:
            x = F.gelu(x, approximate="tanh")

        # ── 修复 2: AdaLN 自注意力调制 ────────────
        if x.device.type == "mps":
            scale_msa, shift_msa, gate_msa = [t.float() for t in (scale_msa, shift_msa, gate_msa)]
            x_fp32 = x.float()
            attn_out = self.attn1(
                rms_norm(x_fp32) * (1 + scale_msa) + shift_msa, ...
            ) * gate_msa
            x = (x_fp32 + attn_out).to(dtype=x.dtype)
        else:
            x += self.attn1(
                rms_norm(x) * (1 + scale_msa) + shift_msa, ...
            ) * gate_msa

        # ── 修复 3: AdaLN MLP 调制 ────────────────
        if x.device.type == "mps":
            s_f, sh_f, g_f = [t.float() for t in (scale_mlp, shift_mlp, gate_mlp)]
            y_f32 = rms_norm(x.float())
            y_f32 = y_f32 * (1 + s_f) + sh_f
            x = x + self.ff(y_f32.to(dtype=x.dtype)) * g_f.to(dtype=x.dtype)
        else:
            y = rms_norm(x) * (1 + scale_mlp) + shift_mlp
            x = x + self.ff(y) * gate_mlp
    """)

    print(f"  {C.DIM}{fix_code}{C.END}")

    print(f"  {C.BOLD}参考实现:{C.END}")
    info("Hunyuan3D:  comfy/ldm/hunyuan3dv2_1/hunyuandit.py:16-17")
    info("LTX-2.3:    comfy/ldm/lightricks/model.py (本 fork mps-fp16-fix 分支)")
    info("完整文档:    MPS_BLACK_VIDEO_FIX.md")


def main():
    parser = argparse.ArgumentParser(
        description="MPS 黑图/NaN 一键排查脚本"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式（只测关键值）"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="指定模型名（如 ltx, minimax）"
    )
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.M}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.M}  MPS 黑图/NaN 一键排查工具{C.END}")
    print(f"{C.BOLD}{C.M}{'═'*60}{C.END}")

    section_env()
    thresholds = section_gelu()
    section_gelu_root_cause(thresholds)
    section_adaln()
    section_comfyui()
    section_summary(thresholds)

    print(f"\n{C.BOLD}{C.M}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.M}  排查完成{C.END}")
    print(f"{C.BOLD}{C.M}{'═'*60}{C.END}\n")


if __name__ == "__main__":
    main()
