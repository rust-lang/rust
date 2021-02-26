use crate::spec::Target;
use crate::spec::crt_objects::new;

pub fn target() -> Target {
    let mut base = super::s390x_unknown_linux_musl::target();

    base.llvm_target = "s390x-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;
    base.options.pre_link_objects_fallback = new(&[]);
    base.options.post_link_objects_fallback = new(&[]);
    base.options.crt_objects_fallback = None;

    base
}
