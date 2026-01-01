use crate::spec::{Cc, LinkerFlavor, Os, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Solaris,
        dynamic_linking: true,
        has_rpath: true,
        families: cvs!["unix"],
        is_like_solaris: true,
        linker_flavor: LinkerFlavor::Unix(Cc::Yes),
        limit_rdylib_exports: false, // Linker doesn't support this
        eh_frame_header: false,

        ..Default::default()
    }
}
