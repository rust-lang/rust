//! This test checks that opaque type collection doesn't try to normalize the projection
//! without respecting its binders (which would ICE).
//! Unfortunately we don't even reach opaque type collection, as we ICE in typeck before that.
//! See #109281 for the original report.
//@ edition:2018
//@ error-pattern: expected generic lifetime parameter, found `'a`

#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

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
    fn make_fut(&self) -> Box<dyn for<'a> Trait<'a, Thing = Fut<'a>>> {
        Box::new((async { () },))
    }
}

fn main() {}
