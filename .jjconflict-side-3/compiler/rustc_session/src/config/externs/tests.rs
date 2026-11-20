use std::path::PathBuf;

use super::split_extern_opt;
use crate::EarlyDiagCtxt;
use crate::config::UnstableOptions;

/// Verifies split_extern_opt handles the supported cases.
#[test]
fn test_split_extern_opt() {
    let early_dcx = EarlyDiagCtxt::new(<_>::default());
    let unstable_opts = &UnstableOptions::default();

    let extern_opt =
        split_extern_opt(&early_dcx, unstable_opts, "priv,noprelude:foo=libbar.rlib").unwrap();
    assert_eq!(extern_opt.crate_name, "foo");
    assert_eq!(extern_opt.path, Some(PathBuf::from("libbar.rlib")));
    assert_eq!(extern_opt.options, Some("priv,noprelude".to_string()));

    let extern_opt = split_extern_opt(&early_dcx, unstable_opts, "priv,noprelude:foo").unwrap();
    assert_eq!(extern_opt.crate_name, "foo");
    assert_eq!(extern_opt.path, None);
    assert_eq!(extern_opt.options, Some("priv,noprelude".to_string()));

    let extern_opt = split_extern_opt(&early_dcx, unstable_opts, "foo=libbar.rlib").unwrap();
    assert_eq!(extern_opt.crate_name, "foo");
    assert_eq!(extern_opt.path, Some(PathBuf::from("libbar.rlib")));
    assert_eq!(extern_opt.options, None);

    let extern_opt = split_extern_opt(&early_dcx, unstable_opts, "foo").unwrap();
    assert_eq!(extern_opt.crate_name, "foo");
    assert_eq!(extern_opt.path, None);
    assert_eq!(extern_opt.options, None);
}

/// Tests some invalid cases for split_extern_opt.
#[test]
fn test_split_extern_opt_invalid() {
    let early_dcx = EarlyDiagCtxt::new(<_>::default());
    let unstable_opts = &UnstableOptions::default();

    // too many `:`s
    let result = split_extern_opt(&early_dcx, unstable_opts, "priv:noprelude:foo=libbar.rlib");
    assert!(result.is_err());
    let _ = result.map_err(|e| e.cancel());

    // can't nest externs without the unstable flag
    let result = split_extern_opt(&early_dcx, unstable_opts, "noprelude:foo::bar=libbar.rlib");
    assert!(result.is_err());
    let _ = result.map_err(|e| e.cancel());
}

/// Tests some cases for split_extern_opt with nested crates like `foo::bar`.
#[test]
fn test_split_extern_opt_nested() {
    let early_dcx = EarlyDiagCtxt::new(<_>::default());
    let unstable_opts = &UnstableOptions { namespaced_crates: true, ..Default::default() };

    let extern_opt =
        split_extern_opt(&early_dcx, unstable_opts, "priv,noprelude:foo::bar=libbar.rlib").unwrap();
    assert_eq!(extern_opt.crate_name, "foo::bar");
    assert_eq!(extern_opt.path, Some(PathBuf::from("libbar.rlib")));
    assert_eq!(extern_opt.options, Some("priv,noprelude".to_string()));

    let extern_opt =
        split_extern_opt(&early_dcx, unstable_opts, "priv,noprelude:foo::bar").unwrap();
    assert_eq!(extern_opt.crate_name, "foo::bar");
    assert_eq!(extern_opt.path, None);
    assert_eq!(extern_opt.options, Some("priv,noprelude".to_string()));

    let extern_opt = split_extern_opt(&early_dcx, unstable_opts, "foo::bar=libbar.rlib").unwrap();
    assert_eq!(extern_opt.crate_name, "foo::bar");
    assert_eq!(extern_opt.path, Some(PathBuf::from("libbar.rlib")));
    assert_eq!(extern_opt.options, None);

    let extern_opt = split_extern_opt(&early_dcx, unstable_opts, "foo::bar").unwrap();
    assert_eq!(extern_opt.crate_name, "foo::bar");
    assert_eq!(extern_opt.path, None);
    assert_eq!(extern_opt.options, None);
}

/// Tests some invalid cases for split_extern_opt with nested crates like `foo::bar`.
#[test]
fn test_split_extern_opt_nested_invalid() {
    let early_dcx = EarlyDiagCtxt::new(<_>::default());
    let unstable_opts = &UnstableOptions { namespaced_crates: true, ..Default::default() };

    // crates can only be nested one deep.
    let result =
        split_extern_opt(&early_dcx, unstable_opts, "priv,noprelude:foo::bar::baz=libbar.rlib");
    assert!(result.is_err());
    let _ = result.map_err(|e| e.cancel());
}
