//! A test for calling `C-unwind` functions across foreign function boundaries.
//!
//! This test triggers a panic when calling a foreign function that calls *back* into Rust.

use std::panic::{AssertUnwindSafe, catch_unwind};

fn main() {
    // Call `add_small_numbers`, passing arguments that will NOT trigger a panic.
    let (a, b) = (9, 1);
    let c = unsafe { add_small_numbers(a, b) };
    assert_eq!(c, 10);

    // Call `add_small_numbers`, passing arguments that will trigger a panic, and catch it.
    let caught_unwind = catch_unwind(AssertUnwindSafe(|| {
        let (a, b) = (10, 1);
        let _c = unsafe { add_small_numbers(a, b) };
        unreachable!("should have unwound instead of returned");
    }));

    // Assert that we did indeed panic, then unwrap and downcast the panic into the sum.
    assert!(caught_unwind.is_err());
    let panic_obj = caught_unwind.unwrap_err();
    let msg = panic_obj.downcast_ref::<String>().unwrap();
    assert_eq!(msg, "11");
}

#[link(name = "add", kind = "static")]
extern "C-unwind" {
    /// An external function, defined in C.
    ///
    /// Returns the sum of two numbers, or panics if the sum is greater than 10.
    fn add_small_numbers(a: u32, b: u32) -> u32;
}

/// This function will panic if `x` is greater than 10.
///
/// This function is called by `add_small_numbers`.
#[no_mangle]
pub extern "C-unwind" fn panic_if_greater_than_10(x: u32) {
    if x > 10 {
        panic!("{}", x); // That is too big!
    }
}
