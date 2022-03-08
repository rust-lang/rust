use crate::spec::{LinkerFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "l4re".to_string(),
        env: "uclibc".to_string(),
        linker_flavor: LinkerFlavor::L4Bender,
        executables: true,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".to_string()),
        linker_is_gnu: false,
        families: vec!["unix".to_string()],
        ..Default::default()
    }
}
