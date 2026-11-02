use crate::spec::{
    Cc, CodeModel, LinkerFlavor, Lld, Os, PanicStrategy, RelocModel, StackProbeType, TargetOptions,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Hyperlight,
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: StackProbeType::Inline,
        code_model: Some(CodeModel::Small),
        relocation_model: RelocModel::Pic,
        position_independent_executables: true,
        static_position_independent_executables: true,
        ..Default::default()
    }
}
