//! Regression test for <https://github.com/rust-lang/rust/issues/28776>.
//! Unsafe fn could be called outside of unsafe block through autoderef.

use std::ptr;

fn main() {
    (&ptr::write)(1 as *mut _, 42);
    //~^ ERROR E0133
}
