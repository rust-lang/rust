// Regression test for 132104

#![feature(coroutine_trait, coroutines)]

use std::ops::Coroutine;
use std::pin::Pin;

fn demo<'not_static>(s: &'not_static str) -> Pin<Box<impl Coroutine<&'not_static str> + 'static>> {
    let mut generator = Box::pin({
        #[coroutine]
        move |ctx: &'not_static str| {
            yield;
            dbg!(ctx);
        }
    });
    generator.as_mut().resume(s);
    generator
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let local = String::from("...");
    let mut coro = demo(&local);
    drop(local);
    let _unrelated = String::from("UAF");
    coro.as_mut().resume("");
}
