//! Regression test for #74400: Type mismatch in function arguments E0631, E0271 are falsely
//! recognized as "implementation of `FnOnce` is not general enough".

use std::convert::identity;

fn main() {}

fn f<T, S>(data: &[T], key: impl Fn(&T) -> S) {
}

fn g<T>(data: &[T]) {
    f(data, identity)
    //~^ ERROR the parameter type
    //~| ERROR the parameter type
    //~| ERROR the parameter type
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
}
