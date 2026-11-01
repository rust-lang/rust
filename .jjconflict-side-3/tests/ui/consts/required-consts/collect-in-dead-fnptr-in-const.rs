//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Late<T>(T);
impl<T> Late<T> {
    const FAIL: () = panic!(); //~ERROR evaluation panicked: explicit panic
    const FNPTR: fn() = || Self::FAIL;
}

// This function is not actually called, but it is mentioned in dead code in a function that is
// called. The function then mentions a const that evaluates to a fnptr that points to a function
// that used a const that fails to evaluate.
// This tests that when processing mentioned items, we also check the fnptrs in the final values
// of consts that we encounter.
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Late::<T>::FNPTR;
    }
}

#[inline(never)]
fn called<T>() {
    if false {
        not_called::<T>();
    }
}

pub fn main() {
    called::<i32>();
}
