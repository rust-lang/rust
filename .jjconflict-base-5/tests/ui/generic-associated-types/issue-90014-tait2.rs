//! This test checks that opaque type collection doesn't try to normalize the projection
//! without respecting its binders (which would ICE).
//! Unfortunately we don't even reach opaque type collection, as we ICE in typeck before that.
//! See #109281 for the original report.
//@ edition:2018
#![feature(type_alias_impl_trait)]

use std::future::Future;

struct Foo<'a>(&'a mut ());

type Fut<'a> = impl Future<Output = ()>;

trait Trait<'x> {
    type Thing;
}

impl<'x, T: 'x> Trait<'x> for (T,) {
    type Thing = T;
}

impl Foo<'_> {
    #[define_opaque(Fut)]
    fn make_fut(&self) -> Box<dyn for<'a> Trait<'a, Thing = Fut<'a>>> {
        Box::new((async { () },))
        //~^ ERROR expected generic lifetime parameter, found `'a`
    }
}

fn main() {}
