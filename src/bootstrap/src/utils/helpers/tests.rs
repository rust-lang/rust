use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use crate::utils::helpers::{
    check_cfg_arg, extract_beta_rev, hex_encode, make, set_file_times, submodule_path_of,
    symlink_dir,
};
use crate::{Config, Flags};

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
fn test_symlink_dir() {
    let config =
        Config::parse(Flags::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]));
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

#[test]
fn test_set_file_times_sanity_check() {
    let config =
        Config::parse(Flags::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]));
    let tempfile = config.tempdir().join(".tmp-file");

    {
        File::create(&tempfile).unwrap().write_all(b"dummy value").unwrap();
        assert!(tempfile.exists());
    }

    // This might only fail on Windows (if file is default read-only then we try to modify file
    // times).
    let unix_epoch = std::time::SystemTime::UNIX_EPOCH;
    let target_time = fs::FileTimes::new().set_accessed(unix_epoch).set_modified(unix_epoch);
    set_file_times(&tempfile, target_time).unwrap();

    let found_metadata = fs::metadata(tempfile).unwrap();
    assert_eq!(found_metadata.accessed().unwrap(), unix_epoch);
    assert_eq!(found_metadata.modified().unwrap(), unix_epoch)
}

#[test]
fn test_submodule_path_of() {
    let config = Config::parse_inner(Flags::parse(&["build".into(), "--dry-run".into()]), |&_| {
        Ok(Default::default())
    });

    let build = crate::Build::new(config.clone());
    let builder = crate::core::builder::Builder::new(&build);
    assert_eq!(submodule_path_of(&builder, "invalid/path"), None);
    assert_eq!(submodule_path_of(&builder, "src/tools/cargo"), Some("src/tools/cargo".to_string()));
    assert_eq!(
        submodule_path_of(&builder, "src/llvm-project"),
        Some("src/llvm-project".to_string())
    );
    // Make sure subdirs are handled properly
    assert_eq!(
        submodule_path_of(&builder, "src/tools/cargo/random-subdir"),
        Some("src/tools/cargo".to_string())
    );
}
