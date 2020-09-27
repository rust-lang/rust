use crate::spec::TargetResult;

pub fn target() -> TargetResult {
    let mut base = super::i686_pc_windows_msvc::target()?;
    base.options.cpu = "pentium".to_string();
    base.llvm_target = "i586-pc-windows-msvc".to_string();
    Ok(base)
}
