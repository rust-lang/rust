use crate::spec::{cvs, PanicStrategy, RelocModel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "linux".into(),
        env: "musl".into(),
        vendor: "unikraft".into(),
        linker: Some("kraftld".into()),
        relocation_model: RelocModel::Static,
        families: cvs!["unix"],
        has_thread_local: true,
        panic_strategy: PanicStrategy::Abort,
        ..Default::default()
    }
}
