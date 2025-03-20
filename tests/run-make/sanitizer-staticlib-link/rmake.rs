// This test first builds a staticlib with AddressSanitizer and checks that
// linking it to an executable fails due to the missing sanitizer runtime.
// It then builds an executable linking to the staticlib and checks that
// the fault in the staticlib is detected correctly.

// Note that checking for the link failure actually checks two things at once:
//   1) That the library has the sanitizer intrumentation
//   2) and that library does not have the sanitizer runtime
// See https://github.com/rust-lang/rust/pull/38699

//@ needs-sanitizer-support
//@ needs-sanitizer-address

//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

use run_make_support::{cc, extra_c_flags, extra_cxx_flags, run_fail, rustc, static_lib_name};

fn main() {
    rustc().arg("-g").arg("-Zsanitizer=address").crate_type("staticlib").input("library.rs").run();
    cc().input("program.c")
        .arg(static_lib_name("library"))
        .out_exe("program")
        .args(extra_c_flags())
        .args(extra_cxx_flags())
        .run_fail();
    rustc().arg("-g").arg("-Zsanitizer=address").crate_type("bin").input("program.rs").run();
    run_fail("program").assert_stderr_contains("stack-buffer-overflow");
}
