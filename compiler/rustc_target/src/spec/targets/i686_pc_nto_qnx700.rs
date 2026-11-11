use crate::spec::base::qnx_sdp;
use crate::spec::{Arch, Env, RustcAbi, StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut meta = qnx_sdp::meta();
    meta.description = Some("32-bit x86 QNX SDP 7.0".into());
    meta.std = Some(false);
    Target {
        llvm_target: "i586-pc-unknown".into(),
        metadata: meta,
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: Arch::X86,
        options: TargetOptions {
            rustc_abi: Some(RustcAbi::X86Sse2),
            cpu: "pentium4".into(),
            max_atomic_width: Some(64),
            pre_link_args: qnx_sdp::pre_link_args(
                qnx_sdp::ApiVariant::Default,
                qnx_sdp::Arch::I586,
            ),
            // for QNX SDP 7.x, we keep target_os = "nto" for backwards compatibility, and use target_env to specify which version
            env: Env::Nto70,
            vendor: "pc".into(),
            stack_probes: StackProbeType::Inline,
            ..base::qnx_sdp::opts()
        },
    }
}
