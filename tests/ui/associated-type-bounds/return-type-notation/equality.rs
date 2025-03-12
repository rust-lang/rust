//@ edition: 2021

use std::future::Future;

trait Trait {
    async fn method() {}
}

fn test<T: Trait<method(..) = Box<dyn Future<Output = ()>>>>() {}
//~^ ERROR return type notation is not allowed to use type equality

fn main() {}
