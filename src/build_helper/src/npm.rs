use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, io};

/// Install all the npm deps, and return the path of `node_modules`.
pub fn install(src_root_path: &Path, out_dir: &Path, yarn: &Path) -> Result<PathBuf, io::Error> {
    let nm_path = out_dir.join("node_modules");
    let copy_to_build = |p| {
        fs::copy(src_root_path.join(p), out_dir.join(p)).map_err(|e| {
            eprintln!("unable to copy {p:?} to build directory: {e:?}");
            e
        })
    };
    // copy stuff to the output directory to make node_modules get put there.
    copy_to_build("package.json")?;
    copy_to_build("yarn.lock")?;

    let mut cmd = Command::new(yarn);
    cmd.arg("install");
    // make sure our `yarn.lock` file actually means something
    cmd.arg("--frozen");

    cmd.current_dir(out_dir);
    let exit_status = cmd
        .spawn()
        .map_err(|err| {
            eprintln!("can not run yarn install");
            io::Error::other(Box::<dyn Error + Send + Sync>::from(format!(
                "unable to run yarn: {}",
                err.kind()
            )))
        })?
        .wait()?;
    if !exit_status.success() {
        eprintln!("yarn install did not exit successfully");
        return Err(io::Error::other(Box::<dyn Error + Send + Sync>::from(format!(
            "yarn install returned exit code {exit_status}"
        ))));
    }
    Ok(nm_path)
}
