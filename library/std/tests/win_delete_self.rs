#![cfg(windows)]

/// Attempting to delete a running binary should return an error on Windows.
#[test]
#[cfg_attr(skip_missing_miri_shims, ignore)] // `remove_file` does not work in Miri on Windows
fn win_delete_self() {
    let path = std::env::current_exe().unwrap();
    assert!(std::fs::remove_file(path).is_err());
}
