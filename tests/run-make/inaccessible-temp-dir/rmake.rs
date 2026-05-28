// Issue #66530: We would ICE if someone compiled with `-o /dev/null`,
// because we would try to generate auxiliary files in `/dev/` (which
// at least the OS X file system rejects).
//
// An attempt to `-Ztemps-dir` into a directory we cannot write into should
// indeed be an error; but not an ICE.
//
// However, some folks run tests as root, which can write `/dev/` and end
// up clobbering `/dev/null`. Instead we'll use an inaccessible path, which
// also used to ICE, but even root can't magically write there.
//
// Note that `-Ztemps-dir` uses `create_dir_all` so it is not sufficient to
// use a directory with non-existing parent like `/does-not-exist/output`.
// See https://github.com/rust-lang/rust/issues/66530

//@ ignore-riscv64
// FIXME: The riscv64gc-gnu build container runs as root, and can always write
// into `inaccessible/tmp`. Ideally, the riscv64-gnu docker container
// would use a non-root user, but this leads to issues with
// `mkfs.ext4 -d`, as well as mounting a loop device for the rootfs.
//@ ignore-arm
// Reason: linker error on `armhf-gnu`
//@ ignore-windows
// Reason: `set_readonly` has no effect on directories
// and does not prevent modification.

use run_make_support::{rfs, rustc, test_while_readonly};

fn main() {
    // Create an inaccessible directory.
    rfs::create_dir("inaccessible");
    test_while_readonly("inaccessible", || {
        // Run rustc with `-Z temps-dir` set to a directory *inside* the inaccessible one,
        // so that it can't create `tmp`.
        rustc()
            .input("program.rs")
            .arg("-Ztemps-dir=inaccessible/tmp")
            .run_fail()
            .assert_stderr_contains(
                "failed to find or create the directory specified by `--temps-dir`",
            );
    });
}
