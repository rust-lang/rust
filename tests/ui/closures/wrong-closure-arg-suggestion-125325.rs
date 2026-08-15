//@ aux-build:wrong-closure-arg-suggestion-aux.rs

// Regression test for #125325

// Tests that we suggest changing an `impl Fn` param
// to `impl FnMut` when the provided closure arg
// is trying to mutate the closure env.
// Ensures that it works that way for both
// functions and methods

extern crate wrong_closure_arg_suggestion_aux as aux;

use aux::{P, PIter, to_fn};

struct S;

impl S {
    fn assoc_func(&self, _f: impl Fn()) -> usize {
        0
    }
}

fn func(_f: impl Fn()) -> usize {
    0
}

fn test_func(s: &S) -> usize {
    let mut x = ();
    s.assoc_func(|| x = ());
    //~^ ERROR cannot assign to `x`, as it is a captured variable in a `Fn` closure
    func(|| x = ())
    //~^ ERROR cannot assign to `x`, as it is a captured variable in a `Fn` closure
}

// Regression test for <https://github.com/rust-lang/rust/issues/155727>.
//
// When the relevant `Fn` bound comes from a non-local callee, we should still
// explain the call-site expectation instead of falling back to the enclosing
// function's return type.
struct Counter {
    counter: i32,
}

impl Counter {
    fn external_fn(mut self) -> i32 {
        take(|_| to_fn(|_| self.counter += 1));
        //~^ ERROR cannot assign to `self.counter`, as `Fn` closures cannot mutate their captured variables [E0594]
        //~| ERROR lifetime may not live long enough
        //~| ERROR cannot borrow `self` as mutable, as it is a captured variable in a `Fn` closure [E0596]
        self.counter
    }

    fn external_method(mut self) -> i32 {
        P.flat_map(|_| P.map(|_| self.counter += 1));
        //~^ ERROR cannot assign to `self.counter`, as `Fn` closures cannot mutate their captured variables [E0594]
        //~| ERROR lifetime may not live long enough
        //~| ERROR cannot borrow `self` as mutable, as it is a captured variable in a `Fn` closure [E0596]
        self.counter
    }
}

fn take<F, U>(_: F)
where
    F: Fn(i32) -> U,
{
}

fn main() {}
