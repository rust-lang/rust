// Crates that are resolved normally have their path canonicalized and all
// symlinks resolved. This did not happen for paths specified
// using the --extern option to rustc, which could lead to rustc thinking
// that it encountered two different versions of a crate, when it's
// actually the same version found through different paths.
// See https://github.com/rust-lang/rust/pull/16505

// This test checks that --extern and symlinks together
// can result in successful compilation.

//@ ignore-cross-compile

use run_make_support::{create_symlink, rustc, tmp_dir};
use std::fs;

fn main() {
    rustc().input("foo.rs").run();
    fs::create_dir_all(tmp_dir().join("other")).unwrap();
    create_symlink(tmp_dir().join("libfoo.rlib"), tmp_dir().join("other"));
    rustc().input("bar.rs").library_search_path(tmp_dir()).run();
    rustc()
        .input("baz.rs")
        .extern_("foo", tmp_dir().join("other/libfoo.rlib"))
        .library_search_path(tmp_dir())
        .run();
}
