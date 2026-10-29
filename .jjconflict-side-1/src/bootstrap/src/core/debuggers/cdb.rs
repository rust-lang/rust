use std::path::PathBuf;

use crate::core::config::TargetSelection;

pub(crate) struct Cdb {
    pub(crate) cdb: PathBuf,
}

/// We consult the registry to find the installed cdb.exe and try "Program Files" if that fails.
#[cfg(windows)]
pub(crate) fn discover_cdb(target: TargetSelection) -> Option<Cdb> {
    if !target.ends_with("-pc-windows-msvc") {
        return None;
    }

    let cdb_arch = if target.starts_with("i686") {
        "x86"
    } else if target.starts_with("x86_64") {
        "x64"
    } else if target.starts_with("aarch64") || target.starts_with("arm64") {
        "arm64"
    } else if target.starts_with("arm") || target.starts_with("thumb") {
        "arm"
    } else {
        return None; // No compatible CDB.exe in the Windows 10 SDK
    };

    let path = discover_cdb_registry(cdb_arch).or_else(|| discover_cdb_program_files(cdb_arch))?;
    Some(Cdb { cdb: path })
}

#[cfg(not(windows))]
pub(crate) fn discover_cdb(_target: TargetSelection) -> Option<Cdb> {
    None
}

#[cfg(windows)]
fn discover_cdb_registry(cdb_arch: &'static str) -> Option<PathBuf> {
    use windows_registry::LOCAL_MACHINE;
    let roots = LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\Windows Kits\Installed Roots").ok()?;
    // "KitsRoot10" is used by both the Windows 10 and 11 SDKs.
    let mut path: PathBuf = roots.get_string("KitsRoot10").ok()?.into();
    path.extend([r"Debuggers", cdb_arch, r"cdb.exe"]);
    path.exists().then_some(path)
}

#[cfg(windows)]
fn discover_cdb_program_files(cdb_arch: &'static str) -> Option<PathBuf> {
    let mut path = PathBuf::from(
        std::env::var_os("ProgramFiles(x86)").or_else(|| std::env::var_os("ProgramFiles"))?,
    );
    path.extend([r"Windows Kits\10\Debuggers", cdb_arch, r"cdb.exe"]);
    path.exists().then_some(path)
}
