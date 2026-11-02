use rustc_abi::Endian;

use crate::spec::{
    Arch, Cc, CfgAbi, LinkerFlavor, Lld, RustcAbi, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "sparc-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("32-bit SPARC Linux".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-i128:128-f128:64-n32-S64".into(),
        arch: Arch::Sparc,
        options: TargetOptions {
            features: "+v8plus".into(),
            rustc_abi: Some(RustcAbi::SparcV8Plus),
            cpu: "v9".into(),
            endian: Endian::Big,
            late_link_args: TargetOptions::link_args(
                LinkerFlavor::Gnu(Cc::Yes, Lld::No),
                &["-mcpu=v9", "-m32"],
            ),
            max_atomic_width: Some(32),
            cfg_abi: CfgAbi::V8Plus,
            ..base::linux_gnu::opts()
        },
    }
}
