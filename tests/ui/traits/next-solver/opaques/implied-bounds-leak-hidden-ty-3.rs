//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// The original regression test for trait-system-refactor-initiative#159.
// Unlike the other tests here the hidden type has a fresh region var which
// makes the implied bound `<T as Trait>::Assoc: 'infer_var`. While this
// variable will end up equal to `'static` later on, we don't really support
// non alias-outlives assumptions with non-universal variables in them. This
// makes this test more involved than the others.

use std::any::Any;

struct Outlives<'a, T>(Option<&'a T>);
trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

// Computing the implied bounds for `foo` normalizes `impl Sized` to
// `Outlives::<'static, <T as Trait>::Assoc>`, adding the implied bound
// `<T as Trait>::Assoc: 'static`.
//
// The caller does not have to prove that bound.
fn foo<T: Trait>(x: <T as Trait>::Assoc) -> (Box<dyn Any>, impl Sized) {
    //[next]~^ ERROR the associated type `<T as Trait>::Assoc` may not live long enough
    (Box::new(x), Outlives::<'static, <T as Trait>::Assoc>(None))
    //~^ ERROR the associated type `<T as Trait>::Assoc` may not live long enough
    //~| ERROR the associated type `<T as Trait>::Assoc` may not live long enough
    //~| ERROR the associated type `<T as Trait>::Assoc` may not live long enough
    //~| ERROR the associated type `<T as Trait>::Assoc` may not live long enough
    //[current]~| ERROR the associated type `<T as Trait>::Assoc` may not live long enough
}

fn main() {
    let string = String::from("temporary");
    let (any, _proof) = foo::<&str>(string.as_str());
    drop(_proof);
    drop(string);
    println!("{}", any.downcast_ref::<&str>().unwrap());
}
