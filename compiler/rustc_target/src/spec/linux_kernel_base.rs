use crate::spec::{PanicStrategy, RelocModel, RelroLevel, StackProbeType, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        env: "gnu".to_string(),
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
        stack_probes: StackProbeType::Call,
        eliminate_frame_pointer: false,
        linker_is_gnu: true,
        position_independent_executables: true,
        needs_plt: true,
        relro_level: RelroLevel::Full,
        relocation_model: RelocModel::Static,

        ..Default::default()
    }
}
