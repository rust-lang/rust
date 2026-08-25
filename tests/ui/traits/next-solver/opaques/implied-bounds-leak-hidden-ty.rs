//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#159.  We need to make sure
// that computing the implied assumptions of `foo` does not look into the hidden
// type of `impl Sized`, as doing so adds a `T: 'static` implied bound which
// its caller does not have to prove.

use std::any::Any;

struct Outlives<T: 'static>(Option<T>);
fn foo<T>(x: T) -> (Box<dyn Any>, impl Sized) {
    //[next]~^ ERROR the parameter type `T` may not live long enough
    (Box::new(x), Outlives::<T>(None))
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //[current]~| ERROR the parameter type `T` may not live long enough
}

fn main() {
    let any = foo(String::from("temporary").as_str()).0;
    println!("{}", any.downcast_ref::<&str>().unwrap());
}
