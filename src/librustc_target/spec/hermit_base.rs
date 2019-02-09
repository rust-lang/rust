use crate::spec::{LinkArgs, LinkerFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![
        "-Wl,-Bstatic".to_string(),
        "-Wl,--no-dynamic-linker".to_string(),
        "-Wl,--gc-sections".to_string(),
        "-Wl,--as-needed".to_string(),
    ]);

    TargetOptions {
        executables: true,
        has_elf_tls: true,
        linker_is_gnu: true,
        no_default_libraries: false,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: false,
        pre_link_args: args,
        relocation_model: "static".to_string(),
        target_family: Some("unix".to_string()),
        tls_model: "local-exec".to_string(),
        .. Default::default()
    }
}
