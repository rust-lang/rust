use crate::spec::{TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "vxworks".into(),
        env: "gnu".into(),
        vendor: "wrs".into(),
        linker: Some("wr-c++".into()),
        exe_suffix: ".vxe".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        has_thread_local: true,
        crt_static_default: true,
        crt_static_respected: true,
        crt_static_allows_dylibs: true,
        // VxWorks needs to implement this to support profiling
        mcount: "_mcount".into(),
        ..Default::default()
    }
}
