use crate::spec::{crt_objects, LinkArgs, LinkOutputKind, LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec![
            "--build-id".to_string(),
            "--eh-frame-hdr".to_string(),
            "--hash-style=gnu".to_string(),
            "-z".to_string(),
            "rodynamic".to_string(),
        ],
    );

    TargetOptions {
        linker: Some("rust-lld".to_owned()),
        lld_flavor: LldFlavor::Ld,
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        is_like_fuchsia: true,
        linker_is_gnu: true,
        has_rpath: false,
        pre_link_args,
        pre_link_objects: crt_objects::new(&[
            (LinkOutputKind::DynamicNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::DynamicPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticPicExe, &["Scrt1.o"]),
        ]),
        position_independent_executables: true,
        has_elf_tls: true,
        ..Default::default()
    }
}
