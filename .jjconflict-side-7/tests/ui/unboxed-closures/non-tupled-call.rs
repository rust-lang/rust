#![feature(fn_traits, unboxed_closures, tuple_trait)]

use std::default::Default;
use std::marker::Tuple;

fn wrap<P: Tuple + Default, T>(func: impl Fn<P, Output = T>) {
    let x: P = Default::default();
    // Should be: `func.call(x);`
    func(x);
    //~^ ERROR cannot use call notation; the first type parameter for the function trait is neither a tuple nor unit
}

fn foo() {}

fn main() {
    wrap(foo);
}
