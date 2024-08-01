#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![allow(dead_code)]

#[derive(PartialEq, Eq)]
struct U;

struct S<const N: U>()
where
    S<{ U }>:;
//~^ ERROR: overflow evaluating the requirement `S<{ U }> well-formed`

fn main() {}
