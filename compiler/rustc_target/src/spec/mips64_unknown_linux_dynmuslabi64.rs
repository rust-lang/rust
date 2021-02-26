use crate::spec::crt_objects::new;
use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::mips64_unknown_linux_muslabi64::target();

    base.llvm_target = "mips64-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;
    base.options.pre_link_objects_fallback = new(&[]);
    base.options.post_link_objects_fallback = new(&[]);
    base.options.crt_objects_fallback = None;

    base
}
