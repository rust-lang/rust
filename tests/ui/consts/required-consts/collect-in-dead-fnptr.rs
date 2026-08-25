//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
}

// This function is not actually called, but it is mentioned in dead code in a function that is
// called. Make sure we still find this error.
// This ensures that we consider ReifyFnPointer coercions when gathering "mentioned" items.
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}

#[inline(never)]
fn called<T>() {
    if false {
        // We don't call the function, but turn it to a function pointer.
        // Make sure it still gest added to `mentioned_items`.
        let _fnptr: fn() = not_called::<T>;
    }
}

pub fn main() {
    called::<i32>();
}
