use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "x86_64-pc-unknown".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("x86 64-bit QNX Neutrino 7.1 RTOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            cpu: "x86-64".into(),
            plt_by_default: false,
            max_atomic_width: Some(64),
            pre_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[
                "-Vgcc_ntox86_64_cxx",
            ]),
            env: "nto71".into(),
            vendor: "pc".into(),
            ..base::nto_qnx::opts()
        },
    }
}
