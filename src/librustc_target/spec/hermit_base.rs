use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, PanicStrategy};
use crate::spec::{RelocModel, TargetOptions, TlsModel};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec!["--build-id".to_string(), "--hash-style=gnu".to_string(), "--Bstatic".to_string()],
    );

    TargetOptions {
        linker: Some("rust-lld".to_owned()),
        executables: true,
        has_elf_tls: true,
        linker_is_gnu: true,
        pre_link_args,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: true,
        relocation_model: RelocModel::Static,
        target_family: None,
        tls_model: TlsModel::InitialExec,
        ..Default::default()
    }
}
