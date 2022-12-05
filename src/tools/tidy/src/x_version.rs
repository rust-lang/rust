use semver::{BuildMetadata, Prerelease, Version};
use std::io::ErrorKind;
use std::process::{Command, Stdio};

pub fn check(bad: &mut bool) {
    let result = Command::new("x")
        .arg("--version")
        .stdout(Stdio::piped())
        .spawn();
    let child = match result {
        Ok(child) => child,
        Err(e) => match e.kind() {
            ErrorKind::NotFound => return (),
            _ => return tidy_error!(bad, "{}", e),
        },
    };

    let output = child.wait_with_output().unwrap();

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
            return tidy_error!(bad, "Current version of x is {version} Consider updating to the newer version of x by running `cargo install --path src/tools/x`");
        }
    } else {
        return tidy_error!(bad, "{}", output.status);
    }
}
