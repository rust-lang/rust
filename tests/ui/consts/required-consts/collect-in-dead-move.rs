//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
}

// This function is not actually called, but is mentioned implicitly as destructor in dead code in a
// function that is called. Make sure we still find this error.
impl<T> Drop for Fail<T> {
    fn drop(&mut self) {
        let _ = Fail::<T>::C;
    }
}

#[inline(never)]
fn called<T>(x: T) {
    if false {
        let v = Fail(x);
        drop(v); // move `v` away (and it then gets dropped there so build still fails)
    }
}

pub fn main() {
    called::<i32>(0);
}
