use crate::spec::Target;

pub(crate) fn target() -> Target {
    let mut base = super::i686_pc_windows_msvc::target();
    base.rustc_abi = None; // overwrite the SSE2 ABI set by the base target
    base.cpu = "pentium".into();
    base.llvm_target = "i586-pc-windows-msvc".into();
    base
}
