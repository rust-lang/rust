// test.c and its static library checkrust.rs make use of variadic functions (VaList).
// This test checks that the use of this feature does not
// prevent the creation of a functional binary.
// See https://github.com/rust-lang/rust/pull/49878

//@ needs-target-std
//@ ignore-android: FIXME(#142855)

use run_make_support::{cc, extra_c_flags, run, rustc, static_lib_name};

fn main() {
    rustc().input("checkrust.rs").run();
    cc().input("test.c")
        .input(static_lib_name("checkrust"))
        .out_exe("test")
        .args(extra_c_flags())
        .run();
    run("test");
}
