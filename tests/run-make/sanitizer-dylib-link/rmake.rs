// This test builds a shared object, then an executable that links it as a native
// rust library (contrast to an rlib). The shared library and executable both
// are compiled with address sanitizer, and we assert that a fault in the dylib
// is correctly detected.
// See https://github.com/rust-lang/rust/pull/38699

//@ needs-sanitizer-support
//@ needs-sanitizer-address

//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

use run_make_support::{run_fail, rustc};

fn main() {
    rustc().arg("-g").arg("-Zsanitizer=address").crate_type("dylib").input("library.rs").run();
    rustc().arg("-g").arg("-Zsanitizer=address").crate_type("bin").input("program.rs").run();
    run_fail("program").assert_stderr_contains("stack-buffer-overflow");
}
