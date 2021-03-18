use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_unknown_linux_gnu::target();
    base.cpu = "i486".to_string();
    base.llvm_target = "i486-unknown-linux-gnu".to_string();
    base
}
