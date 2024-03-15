//@revisions: noopt opt
//@[noopt] build-fail
//@[opt] compile-flags: -O
//FIXME: `opt` revision currently does not stop with an error due to
//<https://github.com/rust-lang/rust/issues/107503>.
//@[opt] build-pass
//! Make sure we detect erroneous constants post-monomorphization even when they are unused. This is
//! crucial, people rely on it for soundness. (https://github.com/rust-lang/rust/issues/112090)

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //[noopt]~ERROR evaluation of `Fail::<i32>::C` failed
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
        // Now it gest dropped implicitly, at the end of this scope.
    }
}

pub fn main() {
    called::<i32>(0);
}
