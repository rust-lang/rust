// This test checks the functionality of foreign function interface (FFI) where Rust
// must call upon a C library defining functions, where these functions also use custom
// types defined by the C file. In addition to compilation being successful, the binary
// should also successfully execute.
// See https://github.com/rust-lang/rust/pull/44295

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("ctest");
    rustc().input("test.rs").run();
    run("test");
}
