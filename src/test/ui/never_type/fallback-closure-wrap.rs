// This is a minified example from Crater breakage observed when attempting to
// stabilize never type, nstoddard/webgl-gui @ 22f0169f.
//
// This particular test case currently fails as the inference to `()` rather
// than `!` happens as a result of an `as` cast, which is not currently tracked.
// Crater did not find many cases of this occuring, but it is included for
// awareness.
//
// check-fail

use std::marker::PhantomData;

fn main() {
    let error = Closure::wrap(Box::new(move || {
        //~^ ERROR type mismatch resolving
        panic!("Can't connect to server.");
    }) as Box<dyn FnMut()>);
}

struct Closure<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Closure<T> {
    fn wrap(data: Box<T>) -> Closure<T> {
        todo!()
    }
}
