// Make sure the workaround for "crate ... required to be available in rlib format, but was not
// found in this form" errors works without `-C prefer-dynamic` (`panic!` calls foreign function
// `__rust_start_panic`).
// no-prefer-dynamic
#![feature(unboxed_closures)]

use std::panic;

#[no_mangle]
extern "C-unwind" fn good_unwind_c() {
    panic!();
}

#[no_mangle]
fn good_unwind_rust() {
    panic!();
}

// Diverging function calls are on a different code path.
#[no_mangle]
extern "rust-call" fn good_unwind_rust_call(_: ()) -> ! {
    panic!();
}

fn main() -> ! {
    extern "C-unwind" {
        fn good_unwind_c();
    }
    panic::catch_unwind(|| unsafe { good_unwind_c() }).unwrap_err();
    extern "Rust" {
        fn good_unwind_rust();
    }
    panic::catch_unwind(|| unsafe { good_unwind_rust() }).unwrap_err();
    extern "rust-call" {
        fn good_unwind_rust_call(_: ()) -> !;
    }
    unsafe { good_unwind_rust_call(()) }
}
