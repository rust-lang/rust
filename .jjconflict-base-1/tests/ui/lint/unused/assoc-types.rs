//@ edition: 2021
//@ revisions: rpitit assoc_ty

#![deny(unused_must_use)]

use std::future::Future;

pub trait Tr {
    type Fut: Future<Output = ()>;

    #[cfg(rpitit)]
    fn foo() -> impl Future<Output = ()>;

    #[cfg(assoc_ty)]
    fn foo() -> Self::Fut;
}

pub async fn bar<T: Tr>() {
    T::foo();
    //~^ ERROR unused implementer of `Future` that must be used
}

fn main() {}
