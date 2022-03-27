use crate::spec::{RelroLevel, TargetOptions};

use super::cvs;

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".into(),
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
