#![feature(c_variadic)]

// Regression test that covers all 3 cases of https://github.com/rust-lang/rust/issues/130372

unsafe extern "C" fn test_va_copy(_: u64, mut ap: ...) {}

pub fn main() {
    unsafe {
        test_va_copy();
        //~^ ERROR this function takes at least 1 argument but 0 arguments were supplied
    }
}
