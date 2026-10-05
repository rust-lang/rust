// Repro for <https://github.com/rust-lang/rust/issues/87425#issue-952059416>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::any::Any;
use std::fmt;
use std::future::Future;

pub trait Foo {
    type Item;
}

impl<F, I> Foo for F
where
    Self: FnOnce() -> I,
    I: fmt::Debug,
{
    type Item = I;
}

async fn foo_item<F: Foo>(_: F) -> F::Item {
    unimplemented!()
}

fn main() {
    let fut = async {
        let callback = || -> Box<dyn Any> { unimplemented!() };

        // Using plain fn instead of a closure fixes the error,
        // though you obviously can't capture any state...
        // fn callback() -> Box<dyn Any> {
        //     todo!()
        // }

        foo_item(callback).await;
    };

    // Removing `+ Send` bound also fixes the error,
    // though at the cost of loosing `Send`ability...
    let fut: &(dyn Future<Output = ()> + Send) = &fut as _;
}
