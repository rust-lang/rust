use super::{RPathConfig};
use super::{minimize_rpaths, rpaths_to_flags, get_rpath_relative_to_output};
use std::path::{Path, PathBuf};

#[test]
fn test_rpaths_to_flags() {
    let flags = rpaths_to_flags(&[
        "path1".to_string(),
        "path2".to_string()
    ]);
    assert_eq!(flags,
               ["-Wl,-rpath,path1",
                "-Wl,-rpath,path2"]);
}

#[test]
fn test_minimize1() {
    let res = minimize_rpaths(&[
        "rpath1".to_string(),
        "rpath2".to_string(),
        "rpath1".to_string()
    ]);
    assert!(res == [
        "rpath1",
        "rpath2",
    ]);
}

#[test]
fn test_minimize2() {
    let res = minimize_rpaths(&[
        "1a".to_string(),
        "2".to_string(),
        "2".to_string(),
        "1a".to_string(),
        "4a".to_string(),
        "1a".to_string(),
        "2".to_string(),
        "3".to_string(),
        "4a".to_string(),
        "3".to_string()
    ]);
    assert!(res == [
        "1a",
        "2",
        "4a",
        "3",
    ]);
}

#[test]
fn test_rpath_relative() {
    if cfg!(target_os = "macos") {
        let config = &mut RPathConfig {
            used_crates: Vec::new(),
            has_rpath: true,
            is_like_osx: true,
            linker_is_gnu: false,
            out_filename: PathBuf::from("bin/rustc"),
            get_install_prefix_lib_path: &mut || panic!(),
        };
        let res = get_rpath_relative_to_output(config,
                                               Path::new("lib/libstd.so"));
        assert_eq!(res, "@loader_path/../lib");
    } else {
        let config = &mut RPathConfig {
            used_crates: Vec::new(),
            out_filename: PathBuf::from("bin/rustc"),
            get_install_prefix_lib_path: &mut || panic!(),
            has_rpath: true,
            is_like_osx: false,
            linker_is_gnu: true,
        };
        let res = get_rpath_relative_to_output(config,
                                               Path::new("lib/libstd.so"));
        assert_eq!(res, "$ORIGIN/../lib");
    }
}

#[test]
fn test_xlinker() {
    let args = rpaths_to_flags(&[
        "a/normal/path".to_string(),
        "a,comma,path".to_string()
    ]);

    assert_eq!(args, vec![
        "-Wl,-rpath,a/normal/path".to_string(),
        "-Wl,-rpath".to_string(),
        "-Xlinker".to_string(),
        "a,comma,path".to_string()
    ]);
}
