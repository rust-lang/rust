use crate::spec::{cvs, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "solaris".into(),
        dynamic_linking: true,
        has_rpath: true,
        families: cvs!["unix"],
        is_like_solaris: true,
        linker_is_gnu: false,
        limit_rdylib_exports: false, // Linker doesn't support this
        eh_frame_header: false,

        ..Default::default()
    }
}
