use super::*;

#[test]
fn test_minimize1() {
    let res = minimize_rpaths(&["rpath1".into(), "rpath2".into(), "rpath1".into()]);
    assert!(res == ["rpath1", "rpath2",]);
}

#[test]
fn test_minimize2() {
    let res = minimize_rpaths(&[
        "1a".into(),
        "2".into(),
        "2".into(),
        "1a".into(),
        "4a".into(),
        "1a".into(),
        "2".into(),
        "3".into(),
        "4a".into(),
        "3".into(),
    ]);
    assert!(res == ["1a", "2", "4a", "3",]);
}

#[test]
fn test_rpath_relative() {
    if cfg!(target_os = "macos") {
        let config = &mut RPathConfig {
            libs: &[],
            is_like_darwin: true,
            linker_is_gnu: false,
            out_filename: PathBuf::from("bin/rustc"),
        };
        let res = get_rpath_relative_to_output(config, Path::new("lib/libstd.so"));
        assert_eq!(res, "@loader_path/../lib");
    } else {
        let config = &mut RPathConfig {
            libs: &[],
            out_filename: PathBuf::from("bin/rustc"),
            is_like_darwin: false,
            linker_is_gnu: true,
        };
        let res = get_rpath_relative_to_output(config, Path::new("lib/libstd.so"));
        assert_eq!(res, "$ORIGIN/../lib");
    }
}

#[test]
fn test_rpath_relative_issue_119571() {
    let config = &mut RPathConfig {
        libs: &[],
        out_filename: PathBuf::from("rustc"),
        is_like_darwin: false,
        linker_is_gnu: true,
    };
    // Should not panic when out_filename only contains filename.
    // Issue 119571
    let _ = get_rpath_relative_to_output(config, Path::new("lib/libstd.so"));
    // Should not panic when lib only contains filename.
    let _ = get_rpath_relative_to_output(config, Path::new("libstd.so"));
}
