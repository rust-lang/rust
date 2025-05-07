use crate::spec::{RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
