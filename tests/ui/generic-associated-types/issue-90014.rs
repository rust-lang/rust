// edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

trait MakeFut {
    type Fut<'a> where Self: 'a;
    fn make_fut<'a>(&'a self) -> Self::Fut<'a>;
}

impl MakeFut for &'_ mut () {
    type Fut<'a> = impl Future<Output = ()>;
    //~^ ERROR: the type `&mut ()` does not fulfill the required lifetime

    fn make_fut<'a>(&'a self) -> Self::Fut<'a> {
        async { () }
    }
}

fn main() {}
