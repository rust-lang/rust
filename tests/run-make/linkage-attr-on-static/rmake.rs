// #[linkage] is a useful attribute which can be applied to statics to allow
// external linkage, something which was not possible before #18890. This test
// checks that using this new feature results in successful compilation and execution.
// See https://github.com/rust-lang/rust/pull/18890

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("foo");
    rustc().input("bar.rs").run();
    run("bar");
}
