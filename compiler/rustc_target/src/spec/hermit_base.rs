use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, TargetOptions, TlsModel};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "hermit".into(),
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
