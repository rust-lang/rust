use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_gnu::target();
    base.cpu = "pentium".into();
    base.llvm_target = "i586-unknown-linux-gnu".into();
    base
}
