#![cfg(windows)]

/// Attempting to delete a running binary should return an error on Windows.
#[test]
fn win_delete_self() {
    let path = std::env::current_exe().unwrap();
    assert!(std::fs::remove_file(path).is_err());
}
