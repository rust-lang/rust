//@ignore-target: windows # File handling is not implemented yet
//@ignore-host: windows # Only supported for UNIX hosts
//@compile-flags: -Zmiri-disable-isolation

#[path = "../utils/mod.rs"]
mod utils;

/// Test that the [`tempfile`] crate is compatible with miri for UNIX hosts and targets
fn main() {
    // Only create a file in our own tmp folder; the "host" temp folder
    // can be nonsensical for cross-tests.
    let dir_path = utils::tmp();
    tempfile::tempfile_in(dir_path).unwrap();
}
