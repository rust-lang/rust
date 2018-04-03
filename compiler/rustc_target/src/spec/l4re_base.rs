use crate::spec::{LinkArgs, LinkerFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "l4re".to_string(),
        env: "uclibc".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        executables: true,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".to_string()),
        pre_link_args: args,
        os_family: Some("unix".to_string()),
        ..Default::default()
    }
}
