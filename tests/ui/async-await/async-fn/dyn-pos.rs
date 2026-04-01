//@ edition:2018

fn foo(x: &dyn AsyncFn()) {}
//~^ ERROR the trait `AsyncFnMut` is not dyn compatible

fn main() {}
