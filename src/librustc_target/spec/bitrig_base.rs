use spec::{TargetOptions, RelroLevel};
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,

        .. Default::default()
    }
}
