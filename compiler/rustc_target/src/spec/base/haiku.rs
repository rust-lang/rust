use crate::spec::{cvs, RelroLevel, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
