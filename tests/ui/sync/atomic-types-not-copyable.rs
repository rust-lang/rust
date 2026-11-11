//! Check that atomic types from `std::sync::atomic` are not `Copy`
//! and cannot be moved out of a shared reference.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/8380>.

use std::ptr;
use std::sync::atomic::*;

fn main() {
    let x = AtomicBool::new(false);
    let x = *&x; //~ ERROR: cannot move out of a shared reference
    let x = AtomicIsize::new(0);
    let x = *&x; //~ ERROR: cannot move out of a shared reference
    let x = AtomicUsize::new(0);
    let x = *&x; //~ ERROR: cannot move out of a shared reference
    let x: AtomicPtr<usize> = AtomicPtr::new(ptr::null_mut());
    let x = *&x; //~ ERROR: cannot move out of a shared reference
}
