// Crates that are resolved normally have their path canonicalized and all
// symlinks resolved. This did not happen for paths specified
// using the --extern option to rustc, which could lead to rustc thinking
// that it encountered two different versions of a crate, when it's
// actually the same version found through different paths.
// See https://github.com/rust-lang/rust/pull/16505

// This test checks that --extern and symlinks together
// can result in successful compilation.

//@ ignore-cross-compile

use run_make_support::{create_symlink, cwd, fs_wrapper, rustc};

fn main() {
    rustc().input("foo.rs").run();
    fs_wrapper::create_dir_all("other");
    create_symlink("libfoo.rlib", "other");
    rustc().input("bar.rs").library_search_path(cwd()).run();
    rustc().input("baz.rs").extern_("foo", "other").library_search_path(cwd()).run();
}
