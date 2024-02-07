#![feature(coroutine_trait)]
#![feature(coroutines)]

use std::ops::Coroutine;

struct Contravariant<'a>(fn(&'a ()));
struct Covariant<'a>(fn() -> &'a ());

fn bad1<'short, 'long: 'short>() -> impl Coroutine<Covariant<'short>> {
    |_: Covariant<'short>| {
        let a: Covariant<'long> = yield ();
        //~^ ERROR lifetime may not live long enough
    }
}

fn bad2<'short, 'long: 'short>() -> impl Coroutine<Contravariant<'long>> {
    |_: Contravariant<'long>| {
        let a: Contravariant<'short> = yield ();
        //~^ ERROR lifetime may not live long enough
    }
}

fn good1<'short, 'long: 'short>() -> impl Coroutine<Covariant<'long>> {
    |_: Covariant<'long>| {
        let a: Covariant<'short> = yield ();
    }
}

fn good2<'short, 'long: 'short>() -> impl Coroutine<Contravariant<'short>> {
    |_: Contravariant<'short>| {
        let a: Contravariant<'long> = yield ();
    }
}

fn main() {}
