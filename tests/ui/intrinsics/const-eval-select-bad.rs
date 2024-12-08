#![feature(const_eval_select)]
#![feature(core_intrinsics)]

use std::intrinsics::const_eval_select;

const fn not_fn_items() {
    const_eval_select((), || {}, || {});
    //~^ ERROR this argument must be a function item
    //~| ERROR this argument must be a function item
    const_eval_select((), 42, 0xDEADBEEF);
    //~^ ERROR expected a `FnOnce()` closure
    //~| ERROR expected a `FnOnce()` closure
    //~| ERROR this argument must be a function item
    //~| ERROR this argument must be a function item
}

const fn foo(n: i32) -> i32 {
    n
}

fn bar(n: i32) -> bool {
    assert_eq!(n, 0, "{} must be equal to {}", n, 0);
    n == 0
}

fn baz(n: bool) -> i32 {
    assert!(n, "{} must be true", n);
    n as i32
}

const fn return_ty_mismatch() {
    const_eval_select((1,), foo, bar);
    //~^ ERROR expected `bar` to be a fn item that returns `i32`, but it returns `bool`
}

const fn args_ty_mismatch() {
    const_eval_select((true,), foo, baz);
    //~^ ERROR type mismatch
}

const fn non_const_fn() {
    const_eval_select((1,), bar, bar);
    //~^ ERROR this argument must be a `const fn`
}

fn main() {}
