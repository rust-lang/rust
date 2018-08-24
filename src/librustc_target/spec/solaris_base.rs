use spec::TargetOptions;
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        dynamic_linking: true,
        executables: true,
        has_rpath: true,
        target_family: Some("unix".to_string()),
        is_like_solaris: true,
        exe_allocation_crate: super::maybe_jemalloc(),

        .. Default::default()
    }
}
