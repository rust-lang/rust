//! Regression test for https://github.com/rust-lang/rust/issues/31511
//! This test confirms that a helpful enough error is shown when you try
//! to incorrectly cast a thin pointer to a fat pointer.

fn cast_thin_to_fat(x: *const ()) {
    x as *const [u8];
    //~^ ERROR: cannot cast thin pointer `*const ()` to wide pointer `*const [u8]`
}

fn main() {}
