//! A test for calling `C-unwind` functions across foreign function boundaries.
//!
//! This test does *not* trigger a panic. The exercises the "happy path" when calling a foreign
//! function that calls *back* into Rust.
#![feature(c_unwind)]

fn main() {
    let (a, b) = (9, 1);
    let c = unsafe { add_small_numbers(a, b) };
    assert_eq!(c, 10);
}

#[link(name = "add", kind = "static")]
extern {
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
        panic!(x); // That is too big!
    }
}
