use crate::spec::{
    Cc, LinkerFlavor, Lld, SanitizerSet, StackProbeType, Target, TargetMetadata, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.stack_probes = StackProbeType::Inline;
    base.static_position_independent_executables = true;
    base.supported_sanitizers = SanitizerSet::ADDRESS
        | SanitizerSet::CFI
        | SanitizerSet::KCFI
        | SanitizerSet::DATAFLOW
        | SanitizerSet::LEAK
        | SanitizerSet::MEMORY
        | SanitizerSet::SAFESTACK
        | SanitizerSet::THREAD;
    base.supports_xray = true;

    // When we're asked to use the `rust-lld` linker by default, set the appropriate lld-using
    // linker flavor, and self-contained linker component.
    if option_env!("CFG_USE_SELF_CONTAINED_LINKER").is_some() {
        base.linker_flavor = LinkerFlavor::Gnu(Cc::Yes, Lld::Yes);
        base.link_self_contained = crate::spec::LinkSelfContainedDefault::with_linker();
    }

    Target {
        llvm_target: "x86_64-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("64-bit Linux (kernel 3.2+, glibc 2.17+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
