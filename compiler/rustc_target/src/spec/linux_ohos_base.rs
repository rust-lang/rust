use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();

    base.env = "ohos".into();
    base.crt_static_default = false;
    base.force_emulated_tls = true;
    base.has_thread_local = false;

    base
}
