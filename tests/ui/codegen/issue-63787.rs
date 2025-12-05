//@ run-pass
//@ compile-flags: -O

// Make sure that `Ref` and `RefMut` do not make false promises about aliasing,
// because once they drop, their reference/pointer can alias other writes.

// Adapted from comex's proof of concept:
// https://github.com/rust-lang/rust/issues/63787#issuecomment-523588164

use std::cell::RefCell;
use std::ops::Deref;

pub fn break_if_r_is_noalias(rc: &RefCell<i32>, r: impl Deref<Target = i32>) -> i32 {
    let ptr1 = &*r as *const i32;
    let a = *r;
    drop(r);
    *rc.borrow_mut() = 2;
    let r2 = rc.borrow();
    let ptr2 = &*r2 as *const i32;
    if ptr2 != ptr1 {
        panic!();
    }
    // If LLVM knows the pointers are the same, and if `r` was `noalias`,
    // then it may replace this with `a + a`, ignoring the earlier write.
    a + *r2
}

fn main() {
    let mut rc = RefCell::new(1);
    let res = break_if_r_is_noalias(&rc, rc.borrow());
    assert_eq!(res, 3);

    *rc.get_mut() = 1;
    let res = break_if_r_is_noalias(&rc, rc.borrow_mut());
    assert_eq!(res, 3);
}
