use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, io};

/// Install an exact package version, and return the path of `node_modules`.
pub fn install_one(
    out_dir: &Path,
    npm_bin: &Path,
    pkg_name: &str,
    pkg_version: &str,
) -> Result<PathBuf, io::Error> {
    let nm_path = out_dir.join("node_modules");
    let _ = fs::create_dir(&nm_path);
    let mut child = Command::new(npm_bin)
        .arg("install")
        .arg("--audit=false")
        .arg("--fund=false")
        .arg(format!("{pkg_name}@{pkg_version}"))
        .current_dir(out_dir)
        .spawn()?;
    let exit_status = child.wait()?;
    if !exit_status.success() {
        eprintln!("npm install did not exit successfully");
        return Err(io::Error::other(Box::<dyn Error + Send + Sync>::from(format!(
            "npm install returned exit code {exit_status}"
        ))));
    }
    Ok(nm_path)
}
