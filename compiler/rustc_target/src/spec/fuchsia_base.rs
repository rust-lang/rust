use crate::spec::{
    crt_objects, cvs, LinkArgs, LinkOutputKind, LinkerFlavor, LldFlavor, TargetOptions,
};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec![
            "--build-id".into(),
            "--hash-style=gnu".into(),
            "-z".into(),
            "max-page-size=4096".into(),
            "-z".into(),
            "now".into(),
            "-z".into(),
            "rodynamic".into(),
            "-z".into(),
            "separate-loadable-segments".into(),
            "--pack-dyn-relocs=relr".into(),
        ],
    );

    TargetOptions {
        os: "fuchsia".into(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        linker: Some("rust-lld".into()),
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        is_like_fuchsia: true,
        pre_link_args,
        pre_link_objects: crt_objects::new(&[
            (LinkOutputKind::DynamicNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::DynamicPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticPicExe, &["Scrt1.o"]),
        ]),
        position_independent_executables: true,
        has_thread_local: true,
        ..Default::default()
    }
}
