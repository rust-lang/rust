use std::path::PathBuf;

use tempfile::TempDir;

use crate::{BuildStamp, Config, Flags};

#[test]
#[should_panic(expected = "prefix can not start or end with '.'")]
fn test_with_invalid_prefix() {
    let dir = TempDir::new().unwrap();
    BuildStamp::new(dir.path()).with_prefix(".invalid");
}

#[test]
#[should_panic(expected = "prefix can not start or end with '.'")]
fn test_with_invalid_prefix2() {
    let dir = TempDir::new().unwrap();
    BuildStamp::new(dir.path()).with_prefix("invalid.");
}

#[test]
fn test_is_up_to_date() {
    let dir = TempDir::new().unwrap();

    let mut build_stamp = BuildStamp::new(dir.path()).add_stamp("v1.0.0");
    build_stamp.write().unwrap();

    assert!(
        build_stamp.is_up_to_date(),
        "Expected stamp file to be up-to-date, but contents do not match the expected value."
    );

    build_stamp.stamp = "dummy value".to_owned();
    assert!(
        !build_stamp.is_up_to_date(),
        "Stamp should no longer be up-to-date as we changed its content right above."
    );

    build_stamp.remove().unwrap();
}

#[test]
fn test_with_prefix() {
    let dir = TempDir::new().unwrap();

    let stamp = BuildStamp::new(dir.path()).add_stamp("v1.0.0");
    assert_eq!(stamp.path.file_name().unwrap(), ".stamp");

    let stamp = stamp.with_prefix("test");
    let expected_filename = ".test-stamp";
    assert_eq!(stamp.path.file_name().unwrap(), expected_filename);

    let stamp = stamp.with_prefix("extra-prefix");
    let expected_filename = ".extra-prefix-test-stamp";
    assert_eq!(stamp.path.file_name().unwrap(), expected_filename);
}
