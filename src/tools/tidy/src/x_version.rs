use semver::{BuildMetadata, Prerelease, Version};
use std::io::ErrorKind;
use std::process::{Command, Stdio};

pub fn check(bad: &mut bool) {
    let result = Command::new("x").arg("--wrapper-version").stdout(Stdio::piped()).spawn();
    // This runs the command inside a temporarily directory.
    // This allows us to compare output of result to see if `--wrapper-version` is not a recognized argument to x.
    let temp_result = Command::new("x")
        .arg("--wrapper-version")
        .current_dir(std::env::temp_dir())
        .stdout(Stdio::piped())
        .spawn();

    let (child, temp_child) = match (result, temp_result) {
        (Ok(child), Ok(temp_child)) => (child, temp_child),
        // what would it mean if the temp cmd error'd?
        (Ok(_child), Err(_e)) => todo!(),
        (Err(e), _) => match e.kind() {
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
        let expected = Version {
            major: 0,
            minor: 1,
            patch: 0,
            pre: Prerelease::new("").unwrap(),
            build: BuildMetadata::EMPTY,
        };
        if version < expected {
            return tidy_error!(
                bad,
                "Current version of x is {version} Consider updating to the newer version of x by running `cargo install --path src/tools/x`"
            );
        }
    } else {
        return tidy_error!(bad, "failed to check version of `x`: {}", output.status);
    }
}
