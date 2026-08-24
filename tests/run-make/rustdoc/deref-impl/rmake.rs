// A very specific set of circumstances (mainly, implementing Deref, and
// having a procedural macro and a Debug derivation in external crates) caused
// an internal compiler error (ICE) when trying to use rustdoc. This test
// reproduces the exact circumstances which caused the bug and checks
// that it does not happen again.
// See https://github.com/rust-lang/rust/issues/38237

//@ ignore-cross-compile

use run_make_support::{cwd, rustc, rustdoc};

fn main() {
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    rustdoc().input("baz.rs").library_search_path(cwd()).out_dir(cwd()).run();
}
