use super::public_extern::*;
use super::*;
use crate::process::Command;

#[test]
fn test_general_available() {
    // Lowest version always available.
    assert_eq!(__isOSVersionAtLeast(0, 0, 0), 1);
    // This high version never available.
    assert_eq!(__isOSVersionAtLeast(9999, 99, 99), 0);
}

#[test]
fn test_saturating() {
    // Higher version than supported by OSVersion -> make sure we saturate.
    assert_eq!(__isOSVersionAtLeast(0x10000, 0, 0), 0);
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "`sw_vers` is only available on host macOS")]
fn compare_against_sw_vers() {
    let sw_vers = Command::new("sw_vers").arg("-productVersion").output().unwrap().stdout;
    let sw_vers = String::from_utf8(sw_vers).unwrap();
    let mut sw_vers = sw_vers.trim().split('.');

    let major: i32 = sw_vers.next().unwrap().parse().unwrap();
    let minor: i32 = sw_vers.next().unwrap_or("0").parse().unwrap();
    let subminor: i32 = sw_vers.next().unwrap_or("0").parse().unwrap();
    assert_eq!(sw_vers.count(), 0);

    // Test directly against the lookup
    assert_eq!(lookup_version().get(), pack_os_version(major as _, minor as _, subminor as _));

    // Current version is available
    assert_eq!(__isOSVersionAtLeast(major, minor, subminor), 1);

    // One lower is available
    assert_eq!(__isOSVersionAtLeast(major, minor, (subminor as u32).saturating_sub(1) as i32), 1);
    assert_eq!(__isOSVersionAtLeast(major, (minor as u32).saturating_sub(1) as i32, subminor), 1);
    assert_eq!(__isOSVersionAtLeast((major as u32).saturating_sub(1) as i32, minor, subminor), 1);

    // One higher isn't available
    assert_eq!(__isOSVersionAtLeast(major, minor, subminor + 1), 0);
    assert_eq!(__isOSVersionAtLeast(major, minor + 1, subminor), 0);
    assert_eq!(__isOSVersionAtLeast(major + 1, minor, subminor), 0);
}

#[test]
fn sysctl_same_as_in_plist() {
    if let Some(version) = version_from_sysctl() {
        assert_eq!(version, version_from_plist());
    }
}

#[test]
fn lookup_idempotent() {
    let version = lookup_version();
    for _ in 0..10 {
        assert_eq!(version, lookup_version());
    }
}

/// Test parsing a bunch of different PLists found in the wild, to ensure that
/// if we decide to parse it without CoreFoundation in the future, that it
/// would continue to work, even on older platforms.
#[test]
fn parse_plist() {
    #[track_caller]
    fn check(
        (major, minor, patch): (u16, u8, u8),
        ios_version: Option<(u16, u8, u8)>,
        plist: &str,
    ) {
        let expected = if cfg!(target_os = "ios") {
            if let Some((ios_major, ios_minor, ios_patch)) = ios_version {
                pack_os_version(ios_major, ios_minor, ios_patch)
            } else if cfg!(target_abi = "macabi") {
                // Skip checking iOS version on Mac Catalyst.
                return;
            } else {
                // iOS version will be parsed from ProductVersion
                pack_os_version(major, minor, patch)
            }
        } else {
            pack_os_version(major, minor, patch)
        };
        let cf_handle = CFHandle::new();
        assert_eq!(expected, parse_version_from_plist(&cf_handle, plist.as_bytes()));
    }

    // macOS 10.3.0
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>ProductBuildVersion</key>
            <string>7B85</string>
            <key>ProductCopyright</key>
            <string>Apple Computer, Inc. 1983-2003</string>
            <key>ProductName</key>
            <string>Mac OS X</string>
            <key>ProductUserVisibleVersion</key>
            <string>10.3</string>
            <key>ProductVersion</key>
            <string>10.3</string>
        </dict>
        </plist>
    "#;
    check((10, 3, 0), None, plist);

    // macOS 10.7.5
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>ProductBuildVersion</key>
            <string>11G63</string>
            <key>ProductCopyright</key>
            <string>1983-2012 Apple Inc.</string>
            <key>ProductName</key>
            <string>Mac OS X</string>
            <key>ProductUserVisibleVersion</key>
            <string>10.7.5</string>
            <key>ProductVersion</key>
            <string>10.7.5</string>
        </dict>
        </plist>
    "#;
    check((10, 7, 5), None, plist);

    // macOS 14.7.4
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>6A558D8A-E2EA-11EF-A1D3-6222CAA672A8</string>
            <key>ProductBuildVersion</key>
            <string>23H420</string>
            <key>ProductCopyright</key>
            <string>1983-2025 Apple Inc.</string>
            <key>ProductName</key>
            <string>macOS</string>
            <key>ProductUserVisibleVersion</key>
            <string>14.7.4</string>
            <key>ProductVersion</key>
            <string>14.7.4</string>
            <key>iOSSupportVersion</key>
            <string>17.7</string>
        </dict>
        </plist>
    "#;
    check((14, 7, 4), Some((17, 7, 0)), plist);

    // SystemVersionCompat.plist on macOS 14.7.4
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>6A558D8A-E2EA-11EF-A1D3-6222CAA672A8</string>
            <key>ProductBuildVersion</key>
            <string>23H420</string>
            <key>ProductCopyright</key>
            <string>1983-2025 Apple Inc.</string>
            <key>ProductName</key>
            <string>Mac OS X</string>
            <key>ProductUserVisibleVersion</key>
            <string>10.16</string>
            <key>ProductVersion</key>
            <string>10.16</string>
            <key>iOSSupportVersion</key>
            <string>17.7</string>
        </dict>
        </plist>
    "#;
    check((10, 16, 0), Some((17, 7, 0)), plist);

    // macOS 15.4 Beta 24E5238a
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>67A50F62-00DA-11F0-BDB6-F99BB8310D2A</string>
            <key>ProductBuildVersion</key>
            <string>24E5238a</string>
            <key>ProductCopyright</key>
            <string>1983-2025 Apple Inc.</string>
            <key>ProductName</key>
            <string>macOS</string>
            <key>ProductUserVisibleVersion</key>
            <string>15.4</string>
            <key>ProductVersion</key>
            <string>15.4</string>
            <key>iOSSupportVersion</key>
            <string>18.4</string>
        </dict>
        </plist>
    "#;
    check((15, 4, 0), Some((18, 4, 0)), plist);

    // iOS Simulator 17.5
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>210B8A2C-09C3-11EF-9DB8-273A64AEFA1C</string>
            <key>ProductBuildVersion</key>
            <string>21F79</string>
            <key>ProductCopyright</key>
            <string>1983-2024 Apple Inc.</string>
            <key>ProductName</key>
            <string>iPhone OS</string>
            <key>ProductVersion</key>
            <string>17.5</string>
        </dict>
        </plist>
    "#;
    check((17, 5, 0), None, plist);

    // visionOS Simulator 2.3
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>57CEFDE6-D079-11EF-837C-8B8C7961D0AC</string>
            <key>ProductBuildVersion</key>
            <string>22N895</string>
            <key>ProductCopyright</key>
            <string>1983-2025 Apple Inc.</string>
            <key>ProductName</key>
            <string>xrOS</string>
            <key>ProductVersion</key>
            <string>2.3</string>
            <key>SystemImageID</key>
            <string>D332C7F1-08DF-4DD9-8122-94EF39A1FB92</string>
            <key>iOSSupportVersion</key>
            <string>18.3</string>
        </dict>
        </plist>
    "#;
    check((2, 3, 0), Some((18, 3, 0)), plist);

    // tvOS Simulator 18.2
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>617587B0-B059-11EF-BE70-4380EDE44645</string>
            <key>ProductBuildVersion</key>
            <string>22K154</string>
            <key>ProductCopyright</key>
            <string>1983-2024 Apple Inc.</string>
            <key>ProductName</key>
            <string>Apple TVOS</string>
            <key>ProductVersion</key>
            <string>18.2</string>
            <key>SystemImageID</key>
            <string>8BB5A425-33F0-4821-9F93-40E7ED92F4E0</string>
        </dict>
        </plist>
    "#;
    check((18, 2, 0), None, plist);

    // watchOS Simulator 11.2
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>BuildID</key>
            <string>BAAE2D54-B122-11EF-BF78-C6C6836B724A</string>
            <key>ProductBuildVersion</key>
            <string>22S99</string>
            <key>ProductCopyright</key>
            <string>1983-2024 Apple Inc.</string>
            <key>ProductName</key>
            <string>Watch OS</string>
            <key>ProductVersion</key>
            <string>11.2</string>
            <key>SystemImageID</key>
            <string>79F773E2-2041-43B4-98EE-FAE52402AE95</string>
        </dict>
        </plist>
    "#;
    check((11, 2, 0), None, plist);

    // iOS 9.3.6
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>ProductBuildVersion</key>
            <string>13G37</string>
            <key>ProductCopyright</key>
            <string>1983-2019 Apple Inc.</string>
            <key>ProductName</key>
            <string>iPhone OS</string>
            <key>ProductVersion</key>
            <string>9.3.6</string>
        </dict>
        </plist>
    "#;
    check((9, 3, 6), None, plist);
}

#[test]
#[should_panic = "SystemVersion.plist did not contain a dictionary at the top level"]
fn invalid_plist() {
    let cf_handle = CFHandle::new();
    let _ = parse_version_from_plist(&cf_handle, b"INVALID");
}

#[test]
#[cfg_attr(
    target_abi = "macabi",
    should_panic = "expected iOSSupportVersion in SystemVersion.plist"
)]
#[cfg_attr(
    not(target_abi = "macabi"),
    should_panic = "expected ProductVersion in SystemVersion.plist"
)]
fn empty_plist() {
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        </dict>
        </plist>
    "#;
    let cf_handle = CFHandle::new();
    let _ = parse_version_from_plist(&cf_handle, plist.as_bytes());
}

#[test]
fn parse_version() {
    #[track_caller]
    fn check(major: u16, minor: u8, patch: u8, version: &str) {
        assert_eq!(
            pack_os_version(major, minor, patch),
            parse_os_version(version.as_bytes()).unwrap()
        )
    }

    check(0, 0, 0, "0");
    check(0, 0, 0, "0.0.0");
    check(1, 0, 0, "1");
    check(1, 2, 0, "1.2");
    check(1, 2, 3, "1.2.3");
    check(9999, 99, 99, "9999.99.99");

    // Check leading zeroes
    check(10, 0, 0, "010");
    check(10, 20, 0, "010.020");
    check(10, 20, 30, "010.020.030");
    check(10000, 100, 100, "000010000.00100.00100");

    // Too many parts
    assert!(parse_os_version(b"1.2.3.4").is_err());

    // Empty
    assert!(parse_os_version(b"").is_err());

    // Invalid digit
    assert!(parse_os_version(b"A.B").is_err());

    // Missing digits
    assert!(parse_os_version(b".").is_err());
    assert!(parse_os_version(b".1").is_err());
    assert!(parse_os_version(b"1.").is_err());

    // Too large
    assert!(parse_os_version(b"100000").is_err());
    assert!(parse_os_version(b"1.1000").is_err());
    assert!(parse_os_version(b"1.1.1000").is_err());
}
