use std::io;
use std::path::PathBuf;
use std::process::Command;

use super::{add_version_to_llvm_target, find_sdk_root, parse_version};

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

fn find_sdk_root_xcrun(sdk_name: &str) -> io::Result<PathBuf> {
    let output = Command::new("xcrun")
        .arg("-sdk")
        .arg(sdk_name.to_lowercase())
        .arg("-show-sdk-path")
        .output()?;
    if output.status.success() {
        // FIXME(madsmtm): If using this for real, we should not error on non-UTF-8 paths.
        let output = std::str::from_utf8(&output.stdout).unwrap();
        Ok(PathBuf::from(output.trim()))
    } else {
        let error = String::from_utf8(output.stderr);
        let error = format!("process exit with error: {}", error.unwrap());
        Err(io::Error::new(io::ErrorKind::Other, error))
    }
}

/// Ensure that our `find_sdk_root` matches `xcrun`'s behaviour.
///
/// `xcrun` is quite slow the first time it's run after a reboot, so this test may take some time.
#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "xcrun is only available on macOS")]
fn test_find_sdk_root() {
    let sdks = [
        "MacOSX",
        "AppleTVOS",
        "AppleTVSimulator",
        "iPhoneOS",
        "iPhoneSimulator",
        "WatchOS",
        "WatchSimulator",
        "XROS",
        "XRSimulator",
    ];
    for sdk_name in sdks {
        if let Ok(expected) = find_sdk_root_xcrun(sdk_name) {
            // `xcrun` prefers `MacOSX14.0.sdk` over `MacOSX.sdk`, so let's compare canonical paths.
            let expected = std::fs::canonicalize(expected).unwrap();
            let actual = find_sdk_root(sdk_name).unwrap();
            let actual = std::fs::canonicalize(actual).unwrap();
            assert_eq!(expected, actual);
        } else {
            // The macOS SDK must always be findable in Rust's CI.
            //
            // The other SDKs are allowed to not be found in the current developer directory when
            // running this test.
            if sdk_name == "MacOSX" {
                panic!("Could not find macOS SDK with `xcrun -sdk macosx -show-sdk-path`");
            }
        }
    }
}
