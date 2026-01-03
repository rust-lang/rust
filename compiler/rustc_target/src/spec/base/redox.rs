use crate::spec::{Cc, Env, LinkerFlavor, Lld, Os, RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Redox,
        env: Env::Relibc,
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_default: true,
        crt_static_respected: true,
        crt_static_allows_dylibs: false,
        late_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-lgcc"]),
        ..Default::default()
    }
}
