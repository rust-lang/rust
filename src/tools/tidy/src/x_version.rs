use std::process::Command;

pub fn check(_bad: &mut bool) {
    let result = Command::new("x").arg("--version").output();
    let output = match result {
        Ok(output) => output,
        Err(_e) => todo!(),
    };

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        assert_eq!("0.1.0", version.trim_end());
    }
    // FIXME: throw some kind of tidy error when the version of x isn't
    // greater than or equal to the version we'd expect.
    //tidy_error!(bad, "Current version of x is {version} Consider updating to the newer version of x by running `cargo install --path src/tools/x`")
}
