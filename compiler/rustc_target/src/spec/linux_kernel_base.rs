use crate::spec::{
    LinkArgs, LinkerFlavor, PanicStrategy, RelocModel, RelroLevel, StackProbeType, TargetOptions,
};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec!["-Wl,--as-needed".to_string(), "-Wl,-z,noexecstack".to_string()],
    );

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
        pre_link_args,

        ..Default::default()
    }
}
