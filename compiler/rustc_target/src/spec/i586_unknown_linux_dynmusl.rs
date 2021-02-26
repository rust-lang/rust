use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_musl::target();

    base.options.cpu = "pentium4".to_string();
    base.llvm_target = "i586-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;
    base.options.static_position_independent_executables = true;

    base
}
