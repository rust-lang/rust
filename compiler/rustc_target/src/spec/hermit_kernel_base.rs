use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, PanicStrategy};
use crate::spec::{RelocModel, TargetOptions, TlsModel};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec!["--build-id".to_string(), "--hash-style=gnu".to_string(), "--Bstatic".to_string()],
    );

    TargetOptions {
        target_os: "hermit".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        disable_redzone: true,
        linker: Some("rust-lld".to_owned()),
        executables: true,
        has_elf_tls: true,
        linker_is_gnu: true,
        pre_link_args,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: true,
        static_position_independent_executables: true,
        relocation_model: RelocModel::Pic,
        target_family: None,
        tls_model: TlsModel::InitialExec,
        ..Default::default()
    }
}
