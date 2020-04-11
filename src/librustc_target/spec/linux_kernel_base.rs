use crate::spec::{LinkArgs, LinkerFlavor, PanicStrategy, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec!["-Wl,--as-needed".to_string(), "-Wl,-z,noexecstack".to_string()],
    );

    TargetOptions {
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: true,
        eliminate_frame_pointer: false,
        linker_is_gnu: true,
        position_independent_executables: true,
        needs_plt: true,
        relro_level: RelroLevel::Full,
        relocation_model: "static".to_string(),
        target_family: Some("unix".to_string()),
        pre_link_args,

        ..Default::default()
    }
}
