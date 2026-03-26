#[test]
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn test_glibc_version() {
    // This mostly just tests that the weak linkage doesn't panic wildly...
    super::glibc_version();
}

#[test]
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn test_parse_glibc_version() {
    let cases = [
        ("0.0", Some((0, 0))),
        ("01.+2", Some((1, 2))),
        ("3.4.5.six", Some((3, 4))),
        ("1", None),
        ("1.-2", None),
        ("1.foo", None),
        ("foo.1", None),
    ];
    for &(version_str, parsed) in cases.iter() {
        assert_eq!(parsed, super::parse_glibc_version(version_str));
    }
}

// Smoke check `confstr`, do it for several hint values, to ensure our resizing
// logic is correct.
#[test]
#[cfg(all(target_vendor = "apple", not(miri)))]
fn test_confstr() {
    for key in [libc::_CS_DARWIN_USER_TEMP_DIR, libc::_CS_PATH] {
        let value_nohint = super::confstr(key, None).unwrap_or_else(|e| {
            panic!("confstr({key}, None) failed: {e:?}");
        });
        let end = (value_nohint.len() + 1) * 2;
        for hint in 0..end {
            assert_eq!(
                super::confstr(key, Some(hint)).as_deref().ok(),
                Some(&*value_nohint),
                "confstr({key}, Some({hint})) failed",
            );
        }
    }
    // Smoke check that we don't loop forever or something if the input was not valid.
    for hint in [None, Some(0), Some(1)] {
        let hopefully_invalid = 123456789_i32;
        assert!(super::confstr(hopefully_invalid, hint).is_err());
    }
}
