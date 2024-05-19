use crate::spec::{base, TargetOptions, TlsModel};

pub fn opts() -> TargetOptions {
    let mut base = base::linux::opts();

    base.env = "ohos".into();
    base.crt_static_default = false;
    base.tls_model = TlsModel::Emulated;
    base.has_thread_local = false;

    base
}
