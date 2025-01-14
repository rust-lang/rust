//@ edition:2018

fn foo(x: &dyn AsyncFn()) {}
//~^ ERROR the trait `AsyncFnMut` cannot be made into an object

fn main() {}
