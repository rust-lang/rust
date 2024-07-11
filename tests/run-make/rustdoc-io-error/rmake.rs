// This test verifies that rustdoc doesn't ICE when it encounters an IO error
// while generating files. Ideally this would be a rustdoc-ui test, so we could
// verify the error message as well.
//
// It operates by creating a temporary directory and modifying its
// permissions so that it is not writable. We have to take special care to set
// the permissions back to normal so that it's able to be deleted later.

//@ ignore-riscv64
//@ ignore-arm
// FIXME: The riscv64gc-gnu and armhf-gnu build containers run as root,
// and can always write into `inaccessible/tmp`. Ideally, these docker
// containers would use a non-root user, but this leads to issues with
// `mkfs.ext4 -d`, as well as mounting a loop device for the rootfs.
//@ ignore-windows - the `set_readonly` functions doesn't work on folders.

use run_make_support::{path, rustdoc};
use std::fs;

fn main() {
    let out_dir = path("rustdoc-io-error");
    let output = fs::create_dir(&out_dir).unwrap();
    let mut permissions = fs::metadata(&out_dir).unwrap().permissions();
    let original_permissions = permissions.clone();

    permissions.set_readonly(true);
    fs::set_permissions(&out_dir, permissions).unwrap();

    let output = rustdoc().input("foo.rs").output(&out_dir).env("RUST_BACKTRACE", "1").run_fail();

    fs::set_permissions(&out_dir, original_permissions).unwrap();

    output
        .assert_exit_code(1)
        .assert_stderr_contains("error: couldn't generate documentation: Permission denied");
}
