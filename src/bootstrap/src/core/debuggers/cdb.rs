use std::env;
use std::path::PathBuf;

use crate::core::config::TargetSelection;

pub(crate) struct Cdb {
    pub(crate) cdb: PathBuf,
}

/// We consult the registry to find the installed cdb.exe and try "Program Files" if that fails.
pub(crate) fn discover_cdb(target: TargetSelection) -> Option<Cdb> {
    if !cfg!(windows) || !target.ends_with("-pc-windows-msvc") {
        return None;
    }

    let cdb_arch = if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "x86_64") {
        if target.starts_with("i686") { "x86" } else { "x64" }
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else {
        return None; // No compatible CDB.exe in the Windows 10 SDK
    };

    let path = discover_cdb_registry(cdb_arch).or_else(|| discover_cdb_program_files(cdb_arch))?;
    Some(Cdb { cdb: path })
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

#[cfg(not(windows))]
fn discover_cdb_registry(_cdb_arch: &'static str) -> Option<PathBuf> {
    None
}

fn discover_cdb_program_files(cdb_arch: &'static str) -> Option<PathBuf> {
    let mut path =
        PathBuf::from(env::var_os("ProgramFiles(x86)").or_else(|| env::var_os("ProgramFiles"))?);
    path.extend([r"Windows Kits\10\Debuggers", cdb_arch, r"cdb.exe"]);
    path.exists().then_some(path)
}
