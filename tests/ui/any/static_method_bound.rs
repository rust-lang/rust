#![feature(try_as_dyn)]

use std::any::try_as_dyn;

type Payload = Box<i32>;

trait Trait {
    fn as_static(&self) -> &'static Payload
    where
        Self: 'static;
}

impl<'a> Trait for &'a Payload {
    fn as_static(&self) -> &'static Payload
    where
        Self: 'static,
    {
        *self
    }
}

fn main() {
    let storage: Box<Payload> = Box::new(Box::new(1i32));
    let wrong: &'static Payload = extend(&*storage);
    drop(storage);
    println!("{wrong}");
}

fn extend(a: &Payload) -> &'static Payload {
    let b: &(dyn Trait + 'static) = try_as_dyn::<&Payload, dyn Trait + 'static>(&a).unwrap();
    //~^ ERROR: the trait bound `(dyn Trait + 'static): TryAsDynCompatible` is not satisfied
    b.as_static()
}
