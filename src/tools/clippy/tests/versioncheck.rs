#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::single_match_else)]

use std::fs;

#[test]
fn consistent_clippy_crate_versions() {
    fn read_version(path: &str) -> String {
        let contents = fs::read_to_string(path).unwrap_or_else(|e| panic!("error reading `{path}`: {e:?}"));
        contents
            .lines()
            .filter_map(|l| l.split_once('='))
            .find_map(|(k, v)| (k.trim() == "version").then(|| v.trim()))
            .unwrap_or_else(|| panic!("error finding version in `{path}`"))
            .to_string()
    }

    // do not run this test inside the upstream rustc repo:
    // https://github.com/rust-lang/rust-clippy/issues/6683
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }

    let clippy_version = read_version("Cargo.toml");

    let paths = [
        "clippy_config/Cargo.toml",
        "clippy_lints/Cargo.toml",
        "clippy_utils/Cargo.toml",
        "declare_clippy_lint/Cargo.toml",
    ];

    for path in paths {
        assert_eq!(clippy_version, read_version(path), "{path} version differs");
    }
}

#[test]
fn check_that_clippy_has_the_same_major_version_as_rustc() {
    // do not run this test inside the upstream rustc repo:
    // https://github.com/rust-lang/rust-clippy/issues/6683
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }

    // Extract `1.XX` from `0.1.XX [(<commit> <date>)]`
    let clippy_version = env!("PKG_VERSION_STR")
        .split(' ')
        .next()
        .unwrap()
        .strip_prefix("0.")
        .unwrap();
    let rustc_version = String::from_utf8(
        std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .expect("failed to run `rustc --version`")
            .stdout,
    )
    .unwrap();
    // extract `1.XX` from `rustc 1.XX.YY-nightly (<commit> <date>)`
    let rustc_version = rustc_version
        .strip_prefix("rustc ")
        .unwrap()
        .split_once('-')
        .unwrap()
        .0
        .rsplit_once('.')
        .unwrap()
        .0;
    assert_eq!(clippy_version, rustc_version);
}
