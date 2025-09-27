use crate::spec::{SanitizerSet, TargetOptions, TlsModel, base};

pub(crate) fn opts() -> TargetOptions {
    let mut base = base::linux::opts();
    base.os = "android".into();
    base.is_like_android = true;
    base.default_dwarf_version = 2;
    base.tls_model = TlsModel::Emulated;
    base.has_thread_local = false;
    base.supported_sanitizers = SanitizerSet::ADDRESS;
    base.crt_static_respected = true;
    base
}
