// Tests that rustc produces helpful error messages when the input file
// cannot be opened, including specific messages for different error kinds
// and typo suggestions for NotFound errors.
//
// The permission-denied test requires Unix file mode bits and is skipped
// on Windows. It is also skipped on riscv64/arm because those CI runners
// run as root, which bypasses permission restrictions.

//@ ignore-riscv64
//@ ignore-arm
//@ ignore-windows
//@ needs-target-std

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use run_make_support::{rfs, run_in_tmpdir, rustc};

fn main() {
    // 1. NotFound — basic case: no "os error 2" in the output
    run_in_tmpdir(|| {
        rustc()
            .input("fo.rs")
            .run_fail()
            .assert_stderr_contains("couldn't find file `fo.rs`")
            .assert_stderr_not_contains("os error 2");
    });

    // 2. NotFound with typo suggestion: foo.rs exists, compiling fo.rs should suggest foo.rs
    run_in_tmpdir(|| {
        rfs::write("foo.rs", b"fn main() {}");
        rustc()
            .input("fo.rs")
            .run_fail()
            .assert_stderr_contains("couldn't find file `fo.rs`")
            .assert_stderr_contains("you might have meant to open `foo.rs`");
    });

    // 3. PermissionDenied — file exists but is unreadable
    run_in_tmpdir(|| {
        rfs::write("secret.rs", b"fn main() {}");

        let mut perms = rfs::metadata("secret.rs").permissions();
        perms.set_mode(0o000); // no read, write, or execute
        rfs::set_permissions("secret.rs", perms);

        // Run rustc before restoring permissions, store the result
        let output = rustc().input("secret.rs").run_fail();

        // Restore permissions so the tmpdir cleanup can delete the file
        let mut perms = rfs::metadata("secret.rs").permissions();
        perms.set_mode(0o644);
        rfs::set_permissions("secret.rs", perms);

        output.assert_stderr_contains("permission denied when opening file");
    });

    // 4. IsADirectory — path points to a directory, not a file
    run_in_tmpdir(|| {
        rfs::create_dir("mydir.rs");
        rustc().input("mydir.rs").run_fail().assert_stderr_contains("is a directory");
    });
}
