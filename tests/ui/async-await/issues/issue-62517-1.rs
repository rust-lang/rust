// Regression test for #62517. We used to ICE when you had an `async
// fn` with an `impl Trait` return that mentioned a `dyn Bar` with no
// explicit lifetime bound.
//
//@ edition:2018
//@ check-pass

trait FirstTrait {}
trait SecondTrait {
    type Item: ?Sized;
}

async fn foo(x: &str) -> impl SecondTrait<Item = dyn FirstTrait> {
}


impl<T> SecondTrait for T {
    type Item = dyn FirstTrait;
}

fn main() { }
