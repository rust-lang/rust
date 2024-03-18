//@revisions: noopt opt
//@ build-fail
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation of `Fail::<i32>::C` failed
}

fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}

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
