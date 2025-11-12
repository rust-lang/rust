// This is a minified example from Crater breakage observed when attempting to
// stabilize never type, nstoddard/webgl-gui @ 22f0169f.
//
// This particular test case currently fails as the inference to `()` rather
// than `!` happens as a result of an `as` cast, which is not currently tracked.
// Crater did not find many cases of this occurring, but it is included for
// awareness.
//
//@ revisions: nofallback fallback
//@[fallback] edition: 2024
//@[nofallback] check-pass
//@[fallback] check-fail

use std::marker::PhantomData;

fn main() {
    let error = Closure::wrap(Box::new(move || {
        panic!("Can't connect to server.");
        //[fallback]~^ ERROR to return `()`, but it returns `!`
    }) as Box<dyn FnMut()>);
}

struct Closure<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Closure<T> {
    fn wrap(data: Box<T>) -> Closure<T> {
        todo!()
    }
}
