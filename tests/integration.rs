#![cfg(feature = "integration")]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::env;
use std::ffi::OsStr;
use std::process::Command;

#[cfg_attr(feature = "integration", test)]
fn integration_test() {
    let repo_name = env::var("INTEGRATION").expect("`INTEGRATION` var not set");
    let repo_url = format!("https://github.com/{}", repo_name);
    let crate_name = repo_name
        .split('/')
        .nth(1)
        .expect("repo name should have format `<org>/<name>`");

    let mut repo_dir = tempfile::tempdir().expect("couldn't create temp dir").into_path();
    repo_dir.push(crate_name);

    let st = Command::new("git")
        .args([
            OsStr::new("clone"),
            OsStr::new("--depth=1"),
            OsStr::new(&repo_url),
            OsStr::new(&repo_dir),
        ])
        .status()
        .expect("unable to run git");
    assert!(st.success());

    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = std::path::Path::new(&root_dir).join("target");
    let clippy_binary = target_dir.join(env!("PROFILE")).join("cargo-clippy");

    let output = Command::new(clippy_binary)
        .current_dir(repo_dir)
        .env("RUST_BACKTRACE", "full")
        .env("CARGO_TARGET_DIR", target_dir)
        .args([
            "clippy",
            "--all-targets",
            "--all-features",
            "--",
            "--cap-lints",
            "warn",
            "-Wclippy::pedantic",
            "-Wclippy::nursery",
        ])
        .output()
        .expect("unable to run clippy");

    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("internal compiler error") {
        let backtrace_start = stderr
            .find("thread 'rustc' panicked at")
            .expect("start of backtrace not found");
        let backtrace_end = stderr
            .rfind("error: internal compiler error")
            .expect("end of backtrace not found");

        panic!(
            "internal compiler error\nBacktrace:\n\n{}",
            &stderr[backtrace_start..backtrace_end]
        );
    } else if stderr.contains("query stack during panic") {
        panic!("query stack during panic in the output");
    } else if stderr.contains("E0463") {
        // Encountering E0463 (can't find crate for `x`) did _not_ cause the build to fail in the
        // past. Even though it should have. That's why we explicitly panic here.
        // See PR #3552 and issue #3523 for more background.
        panic!("error: E0463");
    } else if stderr.contains("E0514") {
        panic!("incompatible crate versions");
    } else if stderr.contains("failed to run `rustc` to learn about target-specific information") {
        panic!("couldn't find librustc_driver, consider setting `LD_LIBRARY_PATH`");
    } else {
        assert!(
            !stderr.contains("toolchain") || !stderr.contains("is not installed"),
            "missing required toolchain"
        );
    }

    match output.status.code() {
        Some(0) => println!("Compilation successful"),
        Some(code) => eprintln!("Compilation failed. Exit code: {}", code),
        None => panic!("Process terminated by signal"),
    }
}
