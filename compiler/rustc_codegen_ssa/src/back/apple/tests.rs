use super::{OSVersion, add_version_to_llvm_target};

#[test]
fn test_add_version_to_llvm_target() {
    assert_eq!(
        add_version_to_llvm_target("aarch64-apple-macosx", OSVersion::new(10, 14, 1)),
        "aarch64-apple-macosx10.14.1"
    );
    assert_eq!(
        add_version_to_llvm_target("aarch64-apple-ios-simulator", OSVersion::new(16, 1, 0)),
        "aarch64-apple-ios16.1.0-simulator"
    );
}
