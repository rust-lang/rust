// run-pass

#![feature(core_intrinsics)]
use std::intrinsics::is_const_eval;

fn main() {
    const X: bool = unsafe { is_const_eval() };
    let y = unsafe { is_const_eval() };
    assert_eq!(X, true);
    assert_eq!(y, false);
}
