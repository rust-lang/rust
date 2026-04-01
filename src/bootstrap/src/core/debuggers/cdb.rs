use std::env;
use std::path::PathBuf;

use crate::core::config::TargetSelection;

pub(crate) struct Cdb {
    pub(crate) cdb: PathBuf,
}

/// FIXME: This CDB discovery code was very questionable when it was in
/// compiletest, and it's just as questionable now that it's in bootstrap.
pub(crate) fn discover_cdb(target: TargetSelection) -> Option<Cdb> {
    if !cfg!(windows) || !target.ends_with("-pc-windows-msvc") {
        return None;
    }

    let pf86 =
        PathBuf::from(env::var_os("ProgramFiles(x86)").or_else(|| env::var_os("ProgramFiles"))?);
    let cdb_arch = if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else {
        return None; // No compatible CDB.exe in the Windows 10 SDK
    };

    let mut path = pf86;
    path.push(r"Windows Kits\10\Debuggers"); // We could check 8.1 etc. too?
    path.push(cdb_arch);
    path.push(r"cdb.exe");

    if !path.exists() {
        return None;
    }

    Some(Cdb { cdb: path })
}
