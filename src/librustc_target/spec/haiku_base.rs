use crate::spec::{TargetOptions, RelroLevel};
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        dynamic_linking: true,
        executables: true,
        has_rpath: false,
        target_family: Some("unix".to_string()),
        relro_level: RelroLevel::Full,
        linker_is_gnu: true,
        .. Default::default()
    }
}
