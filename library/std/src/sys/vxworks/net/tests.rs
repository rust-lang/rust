use super::*;

#[test]
fn test_res_init() {
    // This mostly just tests that the weak linkage doesn't panic wildly...
    res_init_if_glibc_before_2_26().unwrap();
}

#[test]
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
        assert_eq!(parsed, parse_glibc_version(version_str));
    }
}
