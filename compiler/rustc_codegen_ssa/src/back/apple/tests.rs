use super::{add_version_to_llvm_target, parse_version};

#[test]
fn test_add_version_to_llvm_target() {
    assert_eq!(
        add_version_to_llvm_target("aarch64-apple-macosx", (10, 14, 1)),
        "aarch64-apple-macosx10.14.1"
    );
    assert_eq!(
        add_version_to_llvm_target("aarch64-apple-ios-simulator", (16, 1, 0)),
        "aarch64-apple-ios16.1.0-simulator"
    );
}

#[test]
fn test_parse_version() {
    assert_eq!(parse_version("10"), Ok((10, 0, 0)));
    assert_eq!(parse_version("10.12"), Ok((10, 12, 0)));
    assert_eq!(parse_version("10.12.6"), Ok((10, 12, 6)));
    assert_eq!(parse_version("9999.99.99"), Ok((9999, 99, 99)));
}
