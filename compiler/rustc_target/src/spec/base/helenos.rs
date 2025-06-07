use crate::spec::{PanicStrategy, RelroLevel, StackProbeType, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "helenos".into(),

        dynamic_linking: true,
        // we need the linker to keep libgcc and friends
        no_default_libraries: false,
        has_rpath: true,
        relro_level: RelroLevel::Full,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: StackProbeType::Inline,

        ..Default::default()
    }
}
