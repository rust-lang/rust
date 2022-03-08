#![feature(const_eval_select)]

use std::intrinsics::const_eval_select;

const fn not_fn_items() {
    const_eval_select((), || {}, || {});
    //~^ ERROR the trait bound
    const_eval_select((), 42, 0xDEADBEEF);
    //~^ ERROR the trait bound
    //~| ERROR expected a `FnOnce<()>` closure
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
    //~^ ERROR type mismatch
}

const fn args_ty_mismatch() {
    const_eval_select((true,), foo, baz);
    //~^ ERROR type mismatch
}

fn main() {}
