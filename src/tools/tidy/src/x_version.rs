use semver::Version;
use serde_json::Value;
use std::io::ErrorKind;
use std::process::{Command, Stdio};

pub fn check(bad: &mut bool) {
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

        if let Some(expected) = get_x_wrapper_version() {
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
fn get_x_wrapper_version() -> Option<Version> {
    let cmd = Command::new("cargo")
        .arg("metadata")
        .args(["--no-deps", "--format-version", "1", "--manifest-path", "src/tools/x/Cargo.toml"])
        .stdout(Stdio::piped())
        .spawn();

    let child = match cmd {
        Ok(child) => child,
        Err(e) => {
            println!("failed to get version of `x`: {}", e);
            return None;
        }
    };

    let cargo_output = child.wait_with_output().unwrap();
    let cargo_output_str =
        String::from_utf8(cargo_output.stdout).expect("Unable to parse `src/tools/x/Cargo.toml`");

    let v: Value = serde_json::from_str(&cargo_output_str).unwrap();
    let vesrion_str = &v["packages"][0]["version"].as_str()?;

    Some(Version::parse(vesrion_str).unwrap())
}
