#![cfg_attr(feature = "deny-warnings", deny(warnings))]
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
        "declare_clippy_lint/Cargo.toml",
        "clippy_lints/Cargo.toml",
        "clippy_utils/Cargo.toml",
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

    let clippy_version = rustc_tools_util::get_version_info!();
    let clippy_major = clippy_version.major;
    let clippy_minor = clippy_version.minor;
    let clippy_patch = clippy_version.patch;

    // get the rustc version either from the rustc installed with the toolchain file or from
    // `RUSTC_REAL` if Clippy is build in the Rust repo with `./x.py`.
    let rustc = std::env::var("RUSTC_REAL").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = String::from_utf8(
        std::process::Command::new(rustc)
            .arg("--version")
            .output()
            .expect("failed to run `rustc --version`")
            .stdout,
    )
    .unwrap();
    // extract "1 XX 0" from "rustc 1.XX.0-nightly (<commit> <date>)"
    let vsplit: Vec<&str> = rustc_version
        .split(' ')
        .nth(1)
        .unwrap()
        .split('-')
        .next()
        .unwrap()
        .split('.')
        .collect();
    match vsplit.as_slice() {
        [rustc_major, rustc_minor, _rustc_patch] => {
            // clippy 0.1.XX should correspond to rustc 1.XX.0
            assert_eq!(clippy_major, 0); // this will probably stay the same for a long time
            assert_eq!(
                clippy_minor.to_string(),
                *rustc_major,
                "clippy minor version does not equal rustc major version"
            );
            assert_eq!(
                clippy_patch.to_string(),
                *rustc_minor,
                "clippy patch version does not equal rustc minor version"
            );
            // do not check rustc_patch because when a stable-patch-release is made (like 1.50.2),
            // we don't want our tests failing suddenly
        },
        _ => {
            panic!("Failed to parse rustc version: {vsplit:?}");
        },
    };
}
