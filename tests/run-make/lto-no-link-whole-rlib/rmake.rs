// In order to improve linking performance, entire rlibs will only be linked if a dylib is being
// created. Otherwise, an executable will only link one rlib as usual. Linking will fail in this
// test should this optimization be reverted.
// See https://github.com/rust-lang/rust/pull/31460

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("foo");
    build_native_static_lib("bar");
    rustc().input("lib1.rs").run();
    rustc().input("lib2.rs").run();
    rustc().input("main.rs").arg("-Clto").run();
    run("main");
}
