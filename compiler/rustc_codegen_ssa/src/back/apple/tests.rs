use super::*;

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

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "xcode-select is only available on macOS")]
fn lookup_developer_dir() {
    let _developer_dir = xcode_select_developer_dir().unwrap();
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "xcrun is only available on macOS")]
fn lookup_sdk() {
    let (sdk_path, stderr) = xcrun_show_sdk_path("MacOSX", false).unwrap();
    // Check that the found SDK is valid.
    assert!(sdk_path.join("SDKSettings.plist").exists());
    assert_eq!(stderr, "");

    // Test that the SDK root is a subdir of the developer directory.
    if let Some(developer_dir) = xcode_select_developer_dir() {
        // Only run this test if SDKROOT is not set (otherwise xcrun may look up via. that).
        if std::env::var_os("SDKROOT").is_some() {
            assert!(sdk_path.starts_with(&developer_dir));
        }
    }
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "xcrun is only available on macOS")]
fn lookup_sdk_verbose() {
    let (_, stderr) = xcrun_show_sdk_path("MacOSX", true).unwrap();
    // Newer xcrun versions should emit something like this:
    //
    //     xcrun: note: looking up SDK with 'xcodebuild -sdk macosx -version Path'
    //     xcrun: note: xcrun_db = '/var/.../xcrun_db'
    //     xcrun: note: lookup resolved to: '...'
    //     xcrun: note: database key is: ...
    //
    // Or if the value is already cached, something like this:
    //
    //     xcrun: note: database key is: ...
    //     xcrun: note: lookup resolved in '/var/.../xcrun_db' : '...'
    assert!(
        stderr.contains("xcrun: note: lookup resolved"),
        "stderr should contain lookup note: {stderr}",
    );
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "xcrun is only available on macOS")]
fn try_lookup_invalid_sdk() {
    // As a proxy for testing all the different ways that `xcrun` can fail,
    // test the case where an SDK was not found.
    let err = xcrun_show_sdk_path("invalid", false).unwrap_err();
    let XcrunError::Unsuccessful { stderr, .. } = err else {
        panic!("unexpected error kind: {err:?}");
    };
    // Either one of (depending on if using Command Line Tools or full Xcode):
    // xcrun: error: SDK "invalid" cannot be located
    // xcodebuild: error: SDK "invalid" cannot be located.
    assert!(
        stderr.contains(r#"error: SDK "invalid" cannot be located"#),
        "stderr should contain xcodebuild note: {stderr}",
    );
    assert!(
        stderr.contains("xcrun: error: unable to lookup item 'Path' in SDK 'invalid'"),
        "stderr should contain xcrun note: {stderr}",
    );
}
