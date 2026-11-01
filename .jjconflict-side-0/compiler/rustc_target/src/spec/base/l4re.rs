use crate::spec::{
    Cc, Env, LinkOutputKind, LinkSelfContainedComponents, LinkSelfContainedDefault, LinkerFlavor,
    Os, PanicStrategy, TargetOptions, add_link_args, crt_objects, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    // add ld- and cc-style args
    macro_rules! prepare_args {
        ($($val:expr),+) => {{
            let ld_args = &[$($val),+];
            let cc_args = &[$(concat!("-Wl,", $val)),+];

            let mut ret = TargetOptions::link_args(LinkerFlavor::Unix(Cc::No), ld_args);
            add_link_args(&mut ret, LinkerFlavor::Unix(Cc::Yes), cc_args);
            ret
        }};
    }

    let pre_link_args = prepare_args!("-nostdlib", "-dynamic-linker=rom/libld-l4.so");

    let late_link_args = prepare_args!("-lc", "-lgcc_eh");

    let pre_link_objects_self_contained = crt_objects::new(&[
        (LinkOutputKind::StaticNoPicExe, &["crt1.o", "crti.o", "crtbeginT.o"]),
        (LinkOutputKind::StaticPicExe, &["crt1.p.o", "crti.o", "crtbegin.o"]),
        (LinkOutputKind::DynamicNoPicExe, &["crt1.o", "crti.o", "crtbegin.o"]),
        (LinkOutputKind::DynamicPicExe, &["crt1.s.o", "crti.o", "crtbeginS.o"]),
        (LinkOutputKind::DynamicDylib, &["crti.s.o", "crtbeginS.o"]),
        (LinkOutputKind::StaticDylib, &["crti.s.o", "crtbeginS.o"]),
    ]);

    let post_link_objects_self_contained = crt_objects::new(&[
        (LinkOutputKind::StaticNoPicExe, &["crtendT.o", "crtn.o"]),
        (LinkOutputKind::StaticPicExe, &["crtend.o", "crtn.o"]),
        (LinkOutputKind::DynamicNoPicExe, &["crtend.o", "crtn.o"]),
        (LinkOutputKind::DynamicPicExe, &["crtendS.o", "crtn.o"]),
        (LinkOutputKind::DynamicDylib, &["crtendS.o", "crtn.s.o"]),
        (LinkOutputKind::StaticDylib, &["crtendS.o", "crtn.s.o"]),
    ]);

    TargetOptions {
        os: Os::L4Re,
        env: Env::Uclibc,
        families: cvs!["unix"],
        panic_strategy: PanicStrategy::Abort,
        linker_flavor: LinkerFlavor::Unix(Cc::No),
        dynamic_linking: true,
        position_independent_executables: true,
        has_thread_local: true,
        pre_link_args,
        late_link_args,
        pre_link_objects_self_contained,
        post_link_objects_self_contained,
        link_self_contained: LinkSelfContainedDefault::WithComponents(
            LinkSelfContainedComponents::LIBC | LinkSelfContainedComponents::CRT_OBJECTS,
        ),
        ..Default::default()
    }
}
