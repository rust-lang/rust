use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::i686_pc_windows_msvc::target();
    base.cpu = "pentium".into();
    base.llvm_target = "i586-pc-windows-msvc".into();
    base
}
