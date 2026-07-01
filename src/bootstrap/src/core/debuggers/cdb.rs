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

    if let Some(path) = env::var_os("RUSTC_CDB") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(Cdb { cdb: path });
        }
    }

    let cdb_arch = if target.starts_with("x86_64") {
        "x64"
    } else if target.starts_with("x86") || target.starts_with("i686") || target.starts_with("i586")
    {
        "x86"
    } else if target.starts_with("aarch64") {
        "arm64"
    } else if target.starts_with("arm") {
        "arm"
    } else {
        return None;
    };

    let program_files = [env::var_os("ProgramFiles(x86)"), env::var_os("ProgramFiles")];

    let sdk_versions = ["11", "10", "8.1"];

    for base in program_files.iter().flatten() {
        for version in &sdk_versions {
            let mut path = PathBuf::from(base);
            path.push(format!(r"Windows Kits\{}\Debuggers", version));
            path.push(cdb_arch);
            path.push("cdb.exe");

            if path.exists() {
                return Some(Cdb { cdb: path });
            }
        }
    }

    None
}
