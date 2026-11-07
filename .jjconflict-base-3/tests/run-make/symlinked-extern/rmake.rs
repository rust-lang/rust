// Crates that are resolved normally have their path canonicalized and all symlinks resolved. This
// did not happen for paths specified using the `--extern` option to rustc, which could lead to
// rustc thinking that it encountered two different versions of a crate, when it's actually the same
// version found through different paths.
//
// This test checks that `--extern` and symlinks together can result in successful compilation.
//
// See <https://github.com/rust-lang/rust/pull/16505>.

//@ ignore-cross-compile
//@ needs-symlink

use run_make_support::{cwd, path, rfs, rustc};

fn main() {
    rustc().input("foo.rs").run();
    rfs::create_dir_all("other");
    rfs::symlink_file(path("libfoo.rlib"), path("other").join("libfoo.rlib"));

    rustc().input("bar.rs").library_search_path(cwd()).run();
    rustc().input("baz.rs").extern_("foo", "other/libfoo.rlib").library_search_path(cwd()).run();
}
