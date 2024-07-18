// When using foreign function interface (FFI) with C++, it is possible
// to run into a "double unwind" if either both Rust and C++ run into a panic
// and exception at the same time, or C++ encounters two exceptions. In this case,
// one of the panic unwinds would be leaked and the other would be kept, leading
// to undefined behaviour. After this was fixed in #92911, this test checks that
// the keyword "unreachable" indicative of this bug triggering in this specific context
// does not appear after successfully compiling and executing the program.
// See https://github.com/rust-lang/rust/pull/92911

//@ needs-unwind
// Reason: this test exercises panic unwinding
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib_cxx, run_fail, rustc};

fn main() {
    build_native_static_lib_cxx("foo");
    rustc().input("foo.rs").arg("-lfoo").extra_rs_cxx_flags().run();
    run_fail("foo").assert_stdout_not_contains("unreachable");
}
