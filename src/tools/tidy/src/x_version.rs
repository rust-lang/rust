use std::path::Path;
use std::process::{Command, Stdio};

use semver::Version;

pub fn check(root: &Path, cargo: &Path, bad: &mut bool) {
    let cargo_list = Command::new(cargo).args(["install", "--list"]).stdout(Stdio::piped()).spawn();

    let child = match cargo_list {
        Ok(child) => child,
        Err(e) => return tidy_error!(bad, "failed to run `cargo`: {}", e),
    };

    let cargo_list = child.wait_with_output().unwrap();

    if cargo_list.status.success() {
        let exe_list = String::from_utf8_lossy(&cargo_list.stdout);
        let exe_list = exe_list.lines();

        let mut installed: Option<Version> = None;

        for line in exe_list {
            let mut iter = line.split_whitespace();
            if iter.next() == Some("x") {
                if let Some(version) = iter.next() {
                    // Check this is the rust-lang/rust x tool installation since it should be
                    // installed at a path containing `src/tools/x`.
                    if let Some(path) = iter.next()
                        && path.contains("src/tools/x")
                    {
                        let version = version.strip_prefix("v").unwrap();
                        installed = Some(Version::parse(version).unwrap());
                        break;
                    };
                }
            } else {
                continue;
            }
        }
        // Unwrap the some if x is installed, otherwise return because it's fine if x isn't installed.
        let installed = if let Some(i) = installed { i } else { return };

        if let Some(expected) = get_x_wrapper_version(root, cargo) {
            if installed < expected {
                println!(
                    "Current version of x is {installed}, but the latest version is {expected}\nConsider updating to the newer version of x by running `cargo install --path src/tools/x`"
                )
            }
        } else {
            tidy_error!(
                bad,
                "Unable to parse the latest version of `x` at `src/tools/x/Cargo.toml`"
            )
        }
    } else {
        tidy_error!(bad, "failed to check version of `x`: {}", cargo_list.status)
    }
}

// Parse latest version out of `x` Cargo.toml
fn get_x_wrapper_version(root: &Path, cargo: &Path) -> Option<Version> {
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("src/tools/x/Cargo.toml"))
        .no_deps()
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let mut metadata = t!(cmd.exec());
    metadata.packages.pop().map(|x| x.version)
}
