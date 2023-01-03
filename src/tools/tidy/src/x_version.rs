use semver::Version;
use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, Stdio};

pub fn check(root: &Path, cargo: &Path, bad: &mut bool) {
    let result = Command::new("x").arg("--wrapper-version").stdout(Stdio::piped()).spawn();
    // This runs the command inside a temporary directory.
    // This allows us to compare output of result to see if `--wrapper-version` is not a recognized argument to x.
    let temp_result = Command::new("x")
        .arg("--wrapper-version")
        .current_dir(std::env::temp_dir())
        .stdout(Stdio::piped())
        .spawn();

    let (child, temp_child) = match (result, temp_result) {
        (Ok(child), Ok(temp_child)) => (child, temp_child),
        (Err(e), _) | (_, Err(e)) => match e.kind() {
            ErrorKind::NotFound => return,
            _ => return tidy_error!(bad, "failed to run `x`: {}", e),
        },
    };

    let output = child.wait_with_output().unwrap();
    let temp_output = temp_child.wait_with_output().unwrap();

    if output != temp_output {
        return tidy_error!(
            bad,
            "Current version of x does not support the `--wrapper-version` argument\nConsider updating to the newer version of x by running `cargo install --path src/tools/x`"
        );
    }

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        let version = Version::parse(version.trim_end()).unwrap();

        if let Some(expected) = get_x_wrapper_version(root, cargo) {
            if version < expected {
                return tidy_error!(
                    bad,
                    "Current version of x is {version}, but the latest version is {expected}\nConsider updating to the newer version of x by running `cargo install --path src/tools/x`"
                );
            }
        } else {
            return tidy_error!(
                bad,
                "Unable to parse the latest version of `x` at `src/tools/x/Cargo.toml`"
            );
        }
    } else {
        return tidy_error!(bad, "failed to check version of `x`: {}", output.status);
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
