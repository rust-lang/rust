use crate::spec::{base, Cc, LinkerFlavor, MaybeLazy, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::illumos::opts();
    base.pre_link_args =
        MaybeLazy::lazy(|| TargetOptions::link_args(LinkerFlavor::Unix(Cc::Yes), &["-std=c99"]));
    base.max_atomic_width = Some(128);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI;
    base.features = "+v8a".into();

    Target {
        // LLVM does not currently have a separate illumos target,
        // so we still pass Solaris to it
        llvm_target: "aarch64-unknown-solaris2.11".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
