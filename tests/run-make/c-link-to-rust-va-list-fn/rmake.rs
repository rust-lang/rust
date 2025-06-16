// test.c and its static library checkrust.rs make use of variadic functions (VaList).
// This test checks that the use of this feature does not
// prevent the creation of a functional binary.
// See https://github.com/rust-lang/rust/pull/49878

//@ ignore-none
// Reason: no-std is not supported
//@ ignore-nvptx64-nvidia-cuda
// Reason: can't find crate for 'std'

use run_make_support::{cc, extra_c_flags, run, rustc, static_lib_name, target};

fn main() {
    rustc().input("checkrust.rs").target(target()).run();
    cc().input("test.c")
        .input(static_lib_name("checkrust"))
        .out_exe("test")
        .args(extra_c_flags())
        .run();
    run("test");
}
