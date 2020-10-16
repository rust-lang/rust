use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_musl::target();
    base.options.cpu = "pentium".to_string();
    base.llvm_target = "i586-unknown-linux-musl".to_string();
    base
}
