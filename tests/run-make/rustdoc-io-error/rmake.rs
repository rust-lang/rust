// This test verifies that rustdoc doesn't ICE when it encounters an IO error
// while generating files. Ideally this would be a rustdoc-ui test, so we could
// verify the error message as well.
//
// It operates by creating a temporary directory and modifying its
// permissions so that it is not writable. We have to take special care to set
// the permissions back to normal so that it's able to be deleted later.

use run_make_support::{rustdoc, tmp_dir};
use std::fs;

fn main() {
    let out_dir = tmp_dir().join("rustdoc-io-error");
    let output = fs::create_dir(&out_dir).unwrap();
    let mut permissions = fs::metadata(&out_dir).unwrap().permissions();
    let original_permissions = permissions.clone();
    permissions.set_readonly(true);
    fs::set_permissions(&out_dir, permissions.clone()).unwrap();

    let output = rustdoc().input("foo.rs").output(&out_dir).command_output();

    // Changing back permissions.
    fs::set_permissions(&out_dir, original_permissions).unwrap();

    // Checks that rustdoc failed with the error code 1.
    #[cfg(unix)]
    assert_eq!(output.status.code().unwrap(), 1);
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(stderr.contains("error: couldn't generate documentation: Permission denied"));
}
