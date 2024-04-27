use crate::spec::crt_objects;
use crate::spec::{base, LinkSelfContainedDefault, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = base::linux::opts();

    base.env = "musl".into();
    base.pre_link_objects_self_contained = crt_objects::pre_musl_self_contained();
    base.post_link_objects_self_contained = crt_objects::post_musl_self_contained();
    base.link_self_contained = LinkSelfContainedDefault::InferredForMusl;

    // These targets statically link libc by default
    base.crt_static_default = true;

    base
}
