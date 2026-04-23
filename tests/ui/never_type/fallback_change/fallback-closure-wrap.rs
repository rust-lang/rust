// This is a minified example from Crater breakage observed when attempting to
// stabilize never type, nstoddard/webgl-gui @ 22f0169f.
//
// Crater did not find many cases of this occurring, but it is included for
// awareness.
//
//@ edition 2015..2024

use std::marker::PhantomData;

fn main() {
    let error = Closure::wrap(Box::new(move || {
        panic!("Can't connect to server.");
        //~^ ERROR to return `()`, but it returns `!`
    }) as Box<dyn FnMut()>);
}

struct Closure<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Closure<T> {
    fn wrap(data: Box<T>) -> Closure<T> {
        todo!()
    }
}
