#![feature(fn_delegation)]
#![feature(gca_min_const_items)]

use std::gca;

impl
    gca!({
        //~^ ERROR: expected type, found `gca!()` constant
        fn foo() {}
        reuse foo::<>as bar;
        reuse bar;
        //~^ ERROR: the name `bar` is defined multiple times
    })
{
}

fn main() {}
