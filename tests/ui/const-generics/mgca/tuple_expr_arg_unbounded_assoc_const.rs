#![feature(min_generic_const_args, adt_const_params)]

// Regression test for an ICE in privacy checking while walking the `T` qself
// of `T::ASSOC` inside a tuple const argument.

fn takes_tuple<const N: ()>() {}

fn generic_caller<T, const N: u32>() {
    takes_tuple::<{ (N, T::ASSOC) }>;
    //~^ ERROR expected `()`, found `(N, T::ASSOC)`
}

fn main() {}
