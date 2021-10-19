use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, PanicStrategy, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec!["--build-id".to_string(), "--hash-style=gnu".to_string(), "--Bstatic".to_string()],
    );

    TargetOptions {
        os: "hermit".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        disable_redzone: true,
        linker: Some("rust-lld".to_owned()),
        executables: true,
        pre_link_args,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: true,
        static_position_independent_executables: true,
        ..Default::default()
    }
}
