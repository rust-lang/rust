use crate::spec::{TargetOptions, TlsModel, base};

pub(crate) fn opts() -> TargetOptions {
    let mut base = base::linux::opts();

    base.env = "ohos".into();
    base.crt_static_default = false;
    base.tls_model = TlsModel::Emulated;
    base.has_thread_local = false;

    base
}
