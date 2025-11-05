use crate::spec::{Cc, Env, LinkerFlavor, PanicStrategy, RelocModel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "l4re".into(),
        env: Env::Uclibc,
        linker_flavor: LinkerFlavor::Unix(Cc::No),
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".into()),
        families: cvs!["unix"],
        relocation_model: RelocModel::Static,
        ..Default::default()
    }
}
