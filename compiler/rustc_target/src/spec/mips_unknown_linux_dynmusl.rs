use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::mips_unknown_linux_musl::target();

    base.llvm_target = "mips-unknown-linux-musl".to_string();
    base.options.crt_static_default = false;

    base
}
