use crate::spec::{LinkerFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "l4re".into(),
        env: "uclibc".into(),
        linker_flavor: LinkerFlavor::L4Bender,
        executables: true,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".into()),
        linker_is_gnu: false,
        families: vec!["unix".into()],
        ..Default::default()
    }
}
