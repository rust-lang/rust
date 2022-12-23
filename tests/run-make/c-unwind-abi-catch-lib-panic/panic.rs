#![crate_type = "staticlib"]

/// This function will panic if `x` is greater than 10.
///
/// This function is called by `add_small_numbers`.
#[no_mangle]
pub extern "C-unwind" fn panic_if_greater_than_10(x: u32) {
    if x > 10 {
        panic!("{}", x); // That is too big!
    }
}
