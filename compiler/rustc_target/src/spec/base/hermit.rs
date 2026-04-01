use crate::spec::{Cc, LinkerFlavor, Lld, Os, PanicStrategy, TargetOptions, TlsModel};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Hermit,
        linker: Some("rust-lld".into()),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        tls_model: TlsModel::InitialExec,
        position_independent_executables: true,
        static_position_independent_executables: true,
        has_thread_local: true,
        panic_strategy: PanicStrategy::Abort,
        ..Default::default()
    }
}
