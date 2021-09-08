#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::single_match_else)]

use rustc_tools_util::VersionInfo;

#[test]
fn check_that_clippy_lints_and_clippy_utils_have_the_same_version_as_clippy() {
    // do not run this test inside the upstream rustc repo:
    // https://github.com/rust-lang/rust-clippy/issues/6683
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }

    let clippy_meta = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("could not obtain cargo metadata");

    for krate in &["clippy_lints", "clippy_utils"] {
        let krate_meta = cargo_metadata::MetadataCommand::new()
            .current_dir(std::env::current_dir().unwrap().join(krate))
            .no_deps()
            .exec()
            .expect("could not obtain cargo metadata");
        assert_eq!(krate_meta.packages[0].version, clippy_meta.packages[0].version);
        for package in &clippy_meta.packages[0].dependencies {
            if package.name == *krate {
                assert!(package.req.matches(&krate_meta.packages[0].version));
                break;
            }
        }
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
        std::process::Command::new(&rustc)
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
            panic!("Failed to parse rustc version: {:?}", vsplit);
        },
    };
}
