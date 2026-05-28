// The compiler's rules of alignment for indirectly passed values in a 16-byte aligned argument,
// in a C external function, used to be arbitrary. Unexpected behavior would occasionally occur
// and cause memory corruption. This was fixed in #112157, streamlining the way alignment occurs,
// and this test reproduces the case featured in the issue, checking that it compiles and executes
// successfully.
// See https://github.com/rust-lang/rust/issues/80127

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("test.rs").run();
    run("test");
}
