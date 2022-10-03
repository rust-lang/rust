use crate::spec::TargetOptions;
use crate::spec::{FramePointer, PanicStrategy, RelocModel, RelroLevel, StackProbeType};

pub fn opts() -> TargetOptions {
    TargetOptions {
        env: "gnu".into(),
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: StackProbeType::X86,
        frame_pointer: FramePointer::Always,
        position_independent_executables: true,
        needs_plt: true,
        relro_level: RelroLevel::Full,
        relocation_model: RelocModel::Static,

        ..Default::default()
    }
}
