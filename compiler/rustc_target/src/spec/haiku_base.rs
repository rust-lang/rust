use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".into(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".into()],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
