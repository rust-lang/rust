//@ needs-target-std
//
// In this test, the static library foo is made blank, which used to cause
// a compilation error. As the compiler now returns Ok upon encountering a blank
// staticlib as of #12379, this test checks that compilation is successful despite
// the blank staticlib.
// See https://github.com/rust-lang/rust/pull/12379

use run_make_support::{llvm_ar, rustc, static_lib_name};

fn main() {
    llvm_ar().obj_to_ar().output_input(static_lib_name("foo"), "foo.rs").run();
    llvm_ar().arg("d").output_input(static_lib_name("foo"), "foo.rs").run();
    rustc().input("foo.rs").run();
}
