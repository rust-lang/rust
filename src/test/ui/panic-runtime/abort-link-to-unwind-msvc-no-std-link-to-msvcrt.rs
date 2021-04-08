// run-fail
// compile-flags: -C panic=abort
// aux-build:exit-success-if-unwind-msvc-no-std.rs
// only-msvc
// Test that `no_std` with `panic=abort` under MSVC toolchain
// doesn't cause error when linking to msvcrt.
// We don't run this executable because it will hang in `rust_begin_unwind`

#![no_std]
#![no_main]

extern crate exit_success_if_unwind_msvc_no_std;

#[link(name = "msvcrt")]
extern "C" {}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    exit_success_if_unwind_msvc_no_std::bar(do_panic);
    0
}

fn do_panic() {
    panic!();
}
