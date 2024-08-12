// The rust compiler searches by default for libraries in its current directory,
// but used to have difficulty following symlinks leading to required libraries
// if the real ones were located elsewhere. After this was fixed in #13903, this test
// checks that compilation succeeds through use of the symlink.
// See https://github.com/rust-lang/rust/issues/13890

//@ needs-symlink

use run_make_support::{cwd, path, rfs, rust_lib_name, rustc};

fn main() {
    rfs::create_dir("outdir");
    rustc().input("foo.rs").output(path("outdir").join(rust_lib_name("foo"))).run();
    rfs::create_symlink(path("outdir").join(rust_lib_name("foo")), rust_lib_name("foo"));
    // RUSTC_LOG is used for debugging and is not crucial to the test.
    rustc().env("RUSTC_LOG", "rustc_metadata::loader").input("bar.rs").run();
}
