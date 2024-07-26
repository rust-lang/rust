// Rust exceptions can be foreign (from C code, in this test) or local. Foreign
// exceptions should not be caught, as that can cause undefined behaviour. Instead
// of catching them, #102721 made it so that the binary panics in execution with a helpful message.
// This test checks that the correct message appears and that execution fails when trying to catch
// a foreign exception.
// See https://github.com/rust-lang/rust/issues/102715

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ needs-unwind
// Reason: unwinding panics is exercised in this test

//@ ignore-i686-pc-windows-gnu
// Reason: This test doesn't work on 32-bit MinGW as cdylib has its own copy of unwinder
// so cross-DLL unwinding does not work.

use run_make_support::{run_fail, rustc};

fn main() {
    rustc().input("bar.rs").crate_type("cdylib").run();
    rustc().input("foo.rs").run();
    run_fail("foo").assert_stderr_contains("Rust cannot catch foreign exceptions");
}
