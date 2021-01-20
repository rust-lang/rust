use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_gnu::target();
    base.cpu = "i386".to_string();
    base.llvm_target = "i386-unknown-linux-gnu".to_string();
    base
}
