#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
struct U;

#[derive(ConstParamTy, PartialEq, Eq)]
//~^ ERROR overflow evaluating the requirement `S<U> well-formed`
//~| ERROR overflow evaluating the requirement `S<U> well-formed`

struct S<const N: U>()
where
    S<{ U }>:;
//~^ ERROR overflow evaluating the requirement `S<U> well-formed`
//~| ERROR overflow evaluating the requirement `S<U> well-formed`
//~| ERROR overflow evaluating the requirement `S<U> well-formed`

fn main() {}
