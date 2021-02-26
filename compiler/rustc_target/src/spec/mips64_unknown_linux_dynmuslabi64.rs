use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::mips64_unknown_linux_muslabi64::target();

    base.llvm_target = "mips64-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;
    base.options.static_position_independent_executables = true;

    base
}
