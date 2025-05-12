// Regression test for #62517. We used to ICE when you had an `async
// fn` with an `impl Trait` return that mentioned a `dyn Bar` with no
// explicit lifetime bound.
//
//@ edition:2018
//@ check-pass

trait Object {}

trait Alpha<Param: ?Sized> {}

async fn foo<'a>(_: &'a ()) -> impl Alpha<dyn Object> {}

impl<T> Alpha<dyn Object> for T { }

fn main() { }
