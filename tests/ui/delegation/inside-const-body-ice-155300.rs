#![feature(min_generic_const_args)]
#![feature(fn_delegation)]

use std::gca;

pub struct S<const N: usize>;

impl
    S<
        gca!({
            //~^ ERROR: complex const arguments must be placed inside of a `const` block
            fn foo() {}
            reuse foo::<> as bar;
            reuse bar;
            //~^ ERROR: the name `bar` is defined multiple times
        }),
    >
{
}

fn main() {}
