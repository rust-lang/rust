use std::process::Command;

fn main() {
    // Don't rerun the build script unless it changed.
    println!("cargo:rerun-if-changed=build.rs");

    let git_info = get_output("git", &["log", "-1", "--pretty=format:%H %cs"]);
    let git_info = git_info.as_deref().map(|output| {
        // Rerun if the head commit changes.
        let head_path = get_output("git", &["rev-parse", "--git-path", "HEAD"]).unwrap();
        println!("cargo::rerun-if-changed={}", head_path.trim());

        let (hash, date) = output.split_once(' ').unwrap();
        (&hash[..10], date.trim_end())
    });

    let major: u16 = env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap();
    let minor: u16 = env!("CARGO_PKG_VERSION_MINOR").parse().unwrap();
    let patch: u16 = env!("CARGO_PKG_VERSION_PATCH").parse().unwrap();
    let mut version_str = format!("{major}.{minor}.{patch}");
    if let Some((hash, date)) = git_info {
        version_str.extend([" (", hash, " ", date, ")"]);
    }
    println!("cargo:rustc-env=PKG_VERSION_STR={version_str}");
}

fn get_output(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    output
        .status
        .success()
        .then_some(String::from_utf8(output.stdout).unwrap())
}
