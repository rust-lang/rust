//@ known-bug: #135470
//@ compile-flags: -Copt-level=0
//@ edition: 2021

use std::future::Future;
trait Access {
    type Lister;

    fn list() -> impl Future<Output = Self::Lister> {
        async { todo!() }
    }
}

trait Foo {}
impl Access for dyn Foo {
    type Lister = ();
}

fn main() {
    let svc = async {
        async { <dyn Foo>::list() }.await;
    };
    &svc as &dyn Service;
}

trait UnaryService {
    fn call2() {}
}
trait Unimplemented {}
impl<T: Unimplemented> UnaryService for T {}
struct Wrap<T>(T);
impl<T: Send> UnaryService for Wrap<T> {}

trait Service {
    fn call(&self);
}
impl<T: Send> Service for T {
    fn call(&self) {
        Wrap::<T>::call2();
    }
}
