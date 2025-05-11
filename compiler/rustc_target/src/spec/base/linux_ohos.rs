use crate::spec::{TargetOptions, TlsModel, base};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        env: "ohos".into(),
        crt_static_default: false,
        tls_model: TlsModel::Emulated,
        has_thread_local: false,
        ..base::linux::opts()
    }
}
