use crate::spec::TargetOptions;
use crate::spec::{FramePointer, PanicStrategy, RelocModel, RelroLevel, StackProbeType};

pub fn opts() -> TargetOptions {
    TargetOptions {
        env: "gnu".to_string(),
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
        stack_probes: StackProbeType::Call,
        frame_pointer: FramePointer::Always,
        position_independent_executables: true,
        needs_plt: true,
        relro_level: RelroLevel::Full,
        relocation_model: RelocModel::Static,

        ..Default::default()
    }
}
