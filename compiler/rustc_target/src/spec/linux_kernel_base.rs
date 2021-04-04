use crate::spec::{PanicStrategy, RelocModel, RelroLevel, StackProbeType, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        env: "gnu".to_string(),
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: StackProbeType::InlineOrCall { min_llvm_version_for_inline: (11, 0, 1) },
        eliminate_frame_pointer: false,
        linker_is_gnu: true,
        position_independent_executables: true,
        needs_plt: true,
        relro_level: RelroLevel::Full,
        relocation_model: RelocModel::Static,

        ..Default::default()
    }
}
