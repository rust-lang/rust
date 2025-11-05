use crate::spec::{Env, Os, PanicStrategy, RelocModel, TargetOptions, Vendor, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Linux,
        env: Env::Musl,
        vendor: Vendor::Unikraft,
        linker: Some("kraftld".into()),
        relocation_model: RelocModel::Static,
        families: cvs!["unix"],
        has_thread_local: true,
        panic_strategy: PanicStrategy::Abort,
        ..Default::default()
    }
}
