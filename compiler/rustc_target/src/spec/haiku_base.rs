use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".to_string(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".to_string()],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
