use crate::utils::helpers::{extract_beta_rev, hex_encode, make};
use std::path::PathBuf;

#[test]
fn test_make() {
    for (host, make_path) in vec![
        ("dragonfly", PathBuf::from("gmake")),
        ("netbsd", PathBuf::from("gmake")),
        ("freebsd", PathBuf::from("gmake")),
        ("openbsd", PathBuf::from("gmake")),
        ("linux", PathBuf::from("make")),
        // for checking the default
        ("_", PathBuf::from("make")),
    ] {
        assert_eq!(make(host), make_path);
    }
}

#[cfg(unix)]
#[test]
fn test_absolute_unix() {
    use crate::utils::helpers::absolute_unix;

    // Test an absolute path
    let path = PathBuf::from("/home/user/file.txt");
    assert_eq!(absolute_unix(&path).unwrap(), PathBuf::from("/home/user/file.txt"));

    // Test an absolute path with double leading slashes
    let path = PathBuf::from("//root//file.txt");
    assert_eq!(absolute_unix(&path).unwrap(), PathBuf::from("//root/file.txt"));

    // Test a relative path
    let path = PathBuf::from("relative/path");
    assert_eq!(
        absolute_unix(&path).unwrap(),
        std::env::current_dir().unwrap().join("relative/path")
    );
}

#[test]
fn test_beta_rev_parsing() {
    // single digit revision
    assert_eq!(extract_beta_rev("1.99.9-beta.7 (xxxxxx)"), Some("7".to_string()));
    // multiple digits
    assert_eq!(extract_beta_rev("1.99.9-beta.777 (xxxxxx)"), Some("777".to_string()));
    // nightly channel (no beta revision)
    assert_eq!(extract_beta_rev("1.99.9-nightly (xxxxxx)"), None);
    // stable channel (no beta revision)
    assert_eq!(extract_beta_rev("1.99.9 (xxxxxxx)"), None);
    // invalid string
    assert_eq!(extract_beta_rev("invalid"), None);
}

#[test]
fn test_string_to_hex_encode() {
    let input_string = "Hello, World!";
    let hex_string = hex_encode(input_string);
    assert_eq!(hex_string, "48656c6c6f2c20576f726c6421");
}
