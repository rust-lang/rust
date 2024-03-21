//@revisions: noopt opt
//@build-pass
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This passes without optimizations, so it can (and should) also pass with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
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
        std::mem::forget(v);
        // Now the destructor never gets "mentioned" so this build should *not* fail.
        // IOW, this demonstrates that we are using a post-drop-elab notion of "mentioned".
    }
}

pub fn main() {
    called::<i32>(0);
}
