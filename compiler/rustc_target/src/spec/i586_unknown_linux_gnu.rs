use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_gnu::target();
    base.options.cpu = "pentium".to_string();
    base.llvm_target = "i586-unknown-linux-gnu".to_string();
    base
}
