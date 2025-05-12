//! Regression test for [rustc doesn't handle relative symlinks to libraries
//! #13890](https://github.com/rust-lang/rust/issues/13890).
//!
//! This smoke test checks that for a given library search path `P`:
//!
//! - `rustc` is able to locate a library available via a symlink, where:
//!     - the symlink is under the directory subtree of `P`,
//!     - but the actual library is not (it's in a different directory subtree).
//!
//! For example:
//!
//! ```text
//! actual_dir/
//!     libfoo.rlib
//! symlink_dir/  # $CWD set; rustc -L . bar.rs that depends on foo
//!     libfoo.rlib --> ../actual_dir/libfoo.rlib
//! ```
//!
//! Previously, if `rustc` was invoked with CWD set to `symlink_dir/`, it would fail to traverse the
//! symlink to locate `actual_dir/libfoo.rlib`. This was originally fixed in
//! <https://github.com/rust-lang/rust/pull/13903>.

//@ ignore-cross-compile
//@ needs-symlink

use run_make_support::{bare_rustc, cwd, path, rfs, rust_lib_name};

fn main() {
    let actual_lib_dir = path("actual_lib_dir");
    let symlink_lib_dir = path("symlink_lib_dir");
    rfs::create_dir_all(&actual_lib_dir);
    rfs::create_dir_all(&symlink_lib_dir);

    // NOTE: `bare_rustc` is used because it does not introduce an implicit `-L .` library search
    // flag.
    bare_rustc().input("foo.rs").output(actual_lib_dir.join(rust_lib_name("foo"))).run();

    rfs::symlink_file(
        actual_lib_dir.join(rust_lib_name("foo")),
        symlink_lib_dir.join(rust_lib_name("foo")),
    );

    // Make rustc's $CWD be in the directory containing the symlink-to-lib.
    bare_rustc()
        .current_dir(&symlink_lib_dir)
        .library_search_path(".")
        .input(cwd().join("bar.rs"))
        .run();
}
