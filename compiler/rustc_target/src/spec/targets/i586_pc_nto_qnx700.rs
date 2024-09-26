use crate::spec::{Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "i586-pc-unknown".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("32-bit x86 QNX Neutrino 7.0 RTOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: TargetOptions {
            cpu: "pentium4".into(),
            max_atomic_width: Some(64),
            pre_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[
                "-Vgcc_ntox86_cxx",
            ]),
            env: "nto70".into(),
            stack_probes: StackProbeType::Inline,
            ..base::nto_qnx::opts()
        },
    }
}
