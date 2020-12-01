use crate::spec::crt_objects::{self, CrtObjectsFallback};
use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();

    base.env = "musl".to_string();
    base.pre_link_objects_fallback = crt_objects::pre_musl_fallback();
    base.post_link_objects_fallback = crt_objects::post_musl_fallback();
    base.crt_objects_fallback = Some(CrtObjectsFallback::Musl);

    // These targets statically link libc by default
    base.crt_static_default = true;

    base
}
