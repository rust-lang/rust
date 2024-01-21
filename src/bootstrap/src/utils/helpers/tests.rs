use crate::{
    utils::helpers::{
        check_cfg_arg, extract_beta_rev, hex_encode, make, program_out_of_date, symlink_dir,
    },
    Config,
};
use std::{
    fs::{self, remove_file, File},
    io::Write,
    path::PathBuf,
};

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

#[test]
fn test_check_cfg_arg() {
    assert_eq!(check_cfg_arg("bootstrap", None), "--check-cfg=cfg(bootstrap)");
    assert_eq!(
        check_cfg_arg("target_arch", Some(&["s360"])),
        "--check-cfg=cfg(target_arch,values(\"s360\"))"
    );
    assert_eq!(
        check_cfg_arg("target_os", Some(&["nixos", "nix2"])),
        "--check-cfg=cfg(target_os,values(\"nixos\",\"nix2\"))"
    );
}

#[test]
fn test_program_out_of_date() {
    let config = Config::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]);
    let tempfile = config.tempdir().join(".tmp-stamp-file");
    File::create(&tempfile).unwrap().write_all(b"dummy value").unwrap();
    assert!(tempfile.exists());

    // up-to-date
    assert!(!program_out_of_date(&tempfile, "dummy value"));
    // out-of-date
    assert!(program_out_of_date(&tempfile, ""));

    remove_file(tempfile).unwrap();
}

#[test]
fn test_symlink_dir() {
    let config = Config::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]);
    let tempdir = config.tempdir().join(".tmp-dir");
    let link_path = config.tempdir().join(".tmp-link");

    fs::create_dir_all(&tempdir).unwrap();
    symlink_dir(&config, &tempdir, &link_path).unwrap();

    let link_source = fs::read_link(&link_path).unwrap();
    assert_eq!(link_source, tempdir);

    fs::remove_dir(tempdir).unwrap();

    #[cfg(windows)]
    fs::remove_dir(link_path).unwrap();
    #[cfg(not(windows))]
    fs::remove_file(link_path).unwrap();
}
