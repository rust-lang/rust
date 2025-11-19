use crate::spec::{Cc, Env, LinkerFlavor, Os, PanicStrategy, RelocModel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::L4Re,
        env: Env::Uclibc,
        linker_flavor: LinkerFlavor::Unix(Cc::No),
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".into()),
        families: cvs!["unix"],
        relocation_model: RelocModel::Static,
        ..Default::default()
    }
}
