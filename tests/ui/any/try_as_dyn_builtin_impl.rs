#![feature(try_as_dyn)]
//@ run-fail
//@ revisions: next old
//@[next] compile-flags: -Znext-solver

use std::any::{Any, try_as_dyn};

type Payload = Box<i32>;

trait Outlives<'b>: 'b {}

trait WithLt {
    type Ref;
}
impl<'a> WithLt for dyn for<'b> Outlives<'b> + 'a {
    type Ref = &'a Payload;
}

struct Thing<T: WithLt + ?Sized>(T::Ref);

trait Trait {
    fn get(&self) -> &'static Payload;
}
impl<T> Trait for Thing<T>
where
    T: WithLt + for<'b> Outlives<'b> + ?Sized,
{
    fn get(&self) -> &'static Payload {
        let x: &<T as WithLt>::Ref = &self.0;
        let y: &(dyn Any + 'static) = x;
        let z: &&'static Payload = y.downcast_ref().unwrap();
        *z
    }
}

fn extend<'a>(payload: &'a Payload) -> &'static Payload {
    let thing: Thing<dyn for<'b> Outlives<'b> + 'a> = Thing(payload);
    let dy: &dyn Trait = try_as_dyn(&thing).unwrap();
    dy.get()
}

fn main() {
    let payload: Box<Payload> = Box::new(Box::new(1));
    let wrong: &'static Payload = extend(&*payload);
    drop(payload);
    println!("{wrong}");
}
