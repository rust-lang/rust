// Similar to the `return-non-c-like-enum-from-c` test, where
// the C code is the library, and the Rust code compiles
// into the executable. Once again, enum variants should be treated
// like an union of structs, which should prevent segfaults or
// unexpected results. The only difference with the aforementioned
// test is that the structs are passed into C directly through the
// `tt_add` and `t_add` function calls.
// See https://github.com/rust-lang/rust/issues/68190

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("nonclike.rs").arg("-ltest").run();
    run("nonclike");
}
