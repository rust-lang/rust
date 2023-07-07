use super::{LinkArgs, Cc, Lld, LinkerFlavor, PanicStrategy};
use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        vec![
            "-zmax-page-size=4096".into(),
            "-znow".into(),
            "-znoexecstack".into(),
            "-ztext".into(),
            "--execute-only".into(),
        ],
    );
    pre_link_args.insert(
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        vec![
            "-Wl,-zmax-page-size=4096".into(),
            "-Wl,-znow".into(),
            "-Wl,-znoexecstack".into(),
            "-Wl,-ztext".into(),
        ],
    );

    TargetOptions {
        os: "teeos".into(),
        vendor: "unknown".into(),
        dynamic_linking: true,
        executables: true,
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
