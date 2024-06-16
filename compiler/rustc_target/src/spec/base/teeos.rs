use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let pre_link_args = {
        const LLD_ARGS: &[&str] = &["-zmax-page-size=4096", "-znow", "-ztext", "--execute-only"];
        const CC_ARGS: &[&str] =
            &["-Wl,-zmax-page-size=4096", "-Wl,-znow", "-Wl,-ztext", "-mexecute-only"];
        TargetOptions::link_args_list(&[
            (LinkerFlavor::Gnu(Cc::No, Lld::No), LLD_ARGS),
            (LinkerFlavor::Gnu(Cc::Yes, Lld::No), CC_ARGS),
        ])
    };

    TargetOptions {
        os: "teeos".into(),
        vendor: "unknown".into(),
        dynamic_linking: true,
        linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        // rpath hardcodes -Wl, so it can't be used together with ld.lld.
        // C TAs also don't support rpath, so this is fine.
        has_rpath: false,
        // Note: Setting has_thread_local to true causes an error when
        // loading / dyn-linking the TA
        has_thread_local: false,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        crt_static_respected: true,
        pre_link_args,
        panic_strategy: PanicStrategy::Abort,
        ..Default::default()
    }
}
