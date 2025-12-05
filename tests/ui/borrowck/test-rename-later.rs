//@ compile-flags: -Znext-solver

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
    (Box::new(x), Outlives::<'static, <T as Trait>::Assoc>(None))
}

fn main() {
    let string = String::from("temporary");
    let (any, _proof) = foo::<&str>(string.as_str());
    drop(_proof);
    drop(string);
    println!("{}", any.downcast_ref::<&str>().unwrap());
}