//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
}

#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}

#[inline(never)]
fn callit_not(f: impl Fn()) {
    if false {
        f();
    }
}

fn main() {
    if false {
        callit_not(not_called::<i32>)
    }
}
