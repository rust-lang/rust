// Regression test for #62517. We used to ICE when you had an `async
// fn` with an `impl Trait` return that mentioned a `dyn Bar` with no
// explicit lifetime bound.
//
// edition:2018

#![feature(async_await)]

trait Object {}

trait Alpha<Param> {}

async fn foo<'a>(_: &'a ()) -> impl Alpha<dyn Object> {}
//~^ ERROR not satisfied

fn main() { }
