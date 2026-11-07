// The rust compiler used to try to detect duplicated libraries in
// the linking order and remove the duplicates... but certain edge cases,
// such as the one presented in `foo` and `bar` in this test, demand precise
// control over the link order, including duplicates. As the anti-duplication
// filter was removed, this test should now successfully see main be compiled
// and executed.
// See https://github.com/rust-lang/rust/pull/12688

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("foo");
    build_native_static_lib("bar");
    rustc().input("main.rs").run();
    run("main");
}
