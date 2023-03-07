use crate::spec::{cvs, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
