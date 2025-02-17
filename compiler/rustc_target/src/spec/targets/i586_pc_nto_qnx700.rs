use crate::spec::base::nto_qnx;
use crate::spec::{RustcAbi, StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut meta = nto_qnx::meta();
    meta.description = Some("32-bit x86 QNX Neutrino 7.0 RTOS".into());
    meta.std = Some(false);
    Target {
        llvm_target: "i586-pc-unknown".into(),
        metadata: meta,
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: TargetOptions {
            rustc_abi: Some(RustcAbi::X86Sse2),
            cpu: "pentium4".into(),
            max_atomic_width: Some(64),
            pre_link_args: nto_qnx::pre_link_args(
                nto_qnx::ApiVariant::Default,
                nto_qnx::Arch::I586,
            ),
            env: "nto70".into(),
            vendor: "pc".into(),
            stack_probes: StackProbeType::Inline,
            ..base::nto_qnx::opts()
        },
    }
}
