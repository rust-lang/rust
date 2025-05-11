use crate::spec::{
    Cc, FramePointer, LinkOutputKind, LinkerFlavor, Lld, TargetOptions, crt_objects, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    // This mirrors the linker options provided by clang. We presume lld for
    // now. When using clang as the linker it will supply these options for us,
    // so we only list them for ld/lld.
    //
    // https://github.com/llvm/llvm-project/blob/0419db6b95e246fe9dc90b5795beb77c393eb2ce/clang/lib/Driver/ToolChains/Fuchsia.cpp#L32
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &[
            "--build-id",
            "--hash-style=gnu",
            "-z",
            "max-page-size=4096",
            "-z",
            "now",
            "-z",
            "start-stop-visibility=hidden",
            "-z",
            "rodynamic",
            "-z",
            "separate-loadable-segments",
            "-z",
            "rel",
            "--pack-dyn-relocs=relr",
        ],
    );

    TargetOptions {
        os: "fuchsia".into(),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        dynamic_linking: true,
        families: cvs!["unix"],
        pre_link_args,
        pre_link_objects: crt_objects::new(&[
            (LinkOutputKind::DynamicNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::DynamicPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticNoPicExe, &["Scrt1.o"]),
            (LinkOutputKind::StaticPicExe, &["Scrt1.o"]),
        ]),
        position_independent_executables: true,
        has_thread_local: true,
        frame_pointer: FramePointer::NonLeaf,
        ..Default::default()
    }
}
