// LLVM's link-time-optimization (LTO) is a useful feature added to Rust in response
// to #10741. This test uses this feature with `-C lto` alongside a native C library,
// and checks that compilation and execution is successful.
// See https://github.com/rust-lang/rust/issues/10741

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{cc, extra_c_flags, extra_cxx_flags, run, rustc, static_lib_name};

fn main() {
    rustc().input("foo.rs").arg("-Clto").run();
    cc().input("bar.c")
        .arg(static_lib_name("foo"))
        .out_exe("bar")
        .args(extra_c_flags())
        .args(extra_cxx_flags())
        .run();
    run("bar");
}
