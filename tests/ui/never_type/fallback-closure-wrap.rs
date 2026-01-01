// This is a minified example from Crater breakage observed when attempting to
// stabilize never type, nstoddard/webgl-gui @ 22f0169f.
//
// Crater did not find many cases of this occurring, but it is included for
// awareness.
//
//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//
//@[e2021] check-pass

use std::marker::PhantomData;

fn main() {
    let error = Closure::wrap(Box::new(move || {
        panic!("Can't connect to server.");
        //[e2024]~^ ERROR to return `()`, but it returns `!`
    }) as Box<dyn FnMut()>);
}

struct Closure<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Closure<T> {
    fn wrap(data: Box<T>) -> Closure<T> {
        todo!()
    }
}
