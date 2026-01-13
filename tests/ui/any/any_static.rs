//@ revisions: next old
//@[next] compile-flags: -Znext-solver
#![feature(try_as_dyn)]

use std::any::{Any, try_as_dyn};

type Payload = Box<i32>;

fn main() {
    let storage: Box<Payload> = Box::new(Box::new(1i32));
    let wrong: &'static Payload = extend(&*storage);
    drop(storage);
    println!("{wrong}");
}

fn extend(a: &Payload) -> &'static Payload {
    let b: &(dyn Any + 'static) = try_as_dyn::<&Payload, dyn Any + 'static>(&a).unwrap();
    //~^ ERROR: borrowed data escapes outside of function
    let c: &&'static Payload = b.downcast_ref::<&'static Payload>().unwrap();
    *c
}
