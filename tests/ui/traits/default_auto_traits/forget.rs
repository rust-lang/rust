#![crate_type = "lib"]
#![feature(forget_trait)]
#![feature(negative_impls)]
#![feature(min_specialization)]
use std::marker::Forget;
use std::mem;

struct A;

impl !Forget for A {}

fn forget_not_forget_type() {
    mem::forget(A);
    //~^ ERROR: values of type `A` may not be forgotten
}

fn generic_maybe_forget_fn<T: ?Forget>(t: T) {
    mem::forget(t);
    //~^ ERROR: values of type `T` may not be forgotten
}

fn call_maybe_forget_fn_with_not_forget_type() {
    generic_maybe_forget_fn(A);
}
