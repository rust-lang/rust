use crate::utils::{FileUpdater, UpdateStatus, Version, parse_cargo_package};

use std::fmt::{Display, Write as _};

use clap::ValueEnum;

static CARGO_TOML_FILES: &[&str] = &[
    "clippy_config/Cargo.toml",
    "clippy_lints/Cargo.toml",
    "clippy_utils/Cargo.toml",
    "declare_clippy_lint/Cargo.toml",
    "Cargo.toml",
];

pub fn bump_version(mut version: Version) {
    version.minor += 1;

    let mut updater = FileUpdater::default();
    for file in CARGO_TOML_FILES {
        updater.update_file(file, &mut |_, src, dst| {
            let package = parse_cargo_package(src);
            if package.version_range.is_empty() {
                dst.push_str(src);
                UpdateStatus::Unchanged
            } else {
                dst.push_str(&src[..package.version_range.start]);
                write!(dst, "\"{}\"", version.toml_display()).unwrap();
                dst.push_str(&src[package.version_range.end..]);
                UpdateStatus::from_changed(src.get(package.version_range) != dst.get(package.version_range))
            }
        });
    }
}

#[derive(ValueEnum, Copy, Clone)]
pub enum Branch {
    Stable,
    Beta,
    Main,
}

impl Display for Branch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Branch::Stable => "stable",
            Branch::Beta => "beta",
            Branch::Main => "main",
        })
    }
}

pub fn rustc_clippy_commit(rustc_path: String, branch: Branch) {
    const PUSH_PR_DESCRIPTION: &str = "Subtree update of `rust-clippy`";

    let mut git_fetch = std::process::Command::new("git");
    git_fetch.current_dir(&rustc_path);
    git_fetch.args([
        "fetch",
        "--recurse-submodules=no",
        "https://github.com/rust-lang/rust",
        &branch.to_string(),
    ]);

    git_fetch.status().expect("failed to fetch base commit");

    let mut git_log = std::process::Command::new("git");
    git_log.current_dir(rustc_path);
    git_log.args([
        "log",
        "-1",
        "--merges",
        &format!("--grep=\"{PUSH_PR_DESCRIPTION}\""),
        "FETCH_HEAD",
        "--",
        "src/tools/clippy",
    ]);

    let out = git_log.output().expect("failed to run git log");
    let last_rustup_commit = String::from_utf8_lossy(out.stdout.trim_ascii());

    let commit = last_rustup_commit
        .lines()
        .find(|c| c.contains(PUSH_PR_DESCRIPTION))
        .expect("no commit found")
        .trim()
        .rsplit_once('/')
        .expect("no commit hash found")
        .1;

    println!("{commit}");
}
