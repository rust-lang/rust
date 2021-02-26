use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::aarch64_unknown_linux_musl::target();

    base.llvm_target = "aarch64-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;
    base.options.static_position_independent_executables = true;
    base.options.need_rpath = true;

    base
}
