//@ignore-target: windows # File handling is not implemented yet
//@ignore-host: windows # Only supported for UNIX hosts
//@compile-flags: -Zmiri-disable-isolation

#[path = "../utils/mod.rs"]
mod utils;

/// Test that the [`tempfile`] crate is compatible with miri for UNIX hosts and targets
fn main() {
    test_tempfile();
    test_tempfile_in();
}

fn test_tempfile() {
    tempfile::tempfile().unwrap();
}

fn test_tempfile_in() {
    let dir_path = utils::tmp();
    tempfile::tempfile_in(dir_path).unwrap();
}
