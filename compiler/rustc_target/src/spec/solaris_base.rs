use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions {
        target_os: "solaris".to_string(),
        target_vendor: "sun".to_string(),
        dynamic_linking: true,
        executables: true,
        has_rpath: true,
        target_family: Some("unix".to_string()),
        is_like_solaris: true,
        limit_rdylib_exports: false, // Linker doesn't support this
        eh_frame_header: false,

        ..Default::default()
    }
}
