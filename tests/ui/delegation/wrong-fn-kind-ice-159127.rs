#![feature(fn_delegation)]
#![feature(min_generic_const_args)]

impl
    core::direct_const_arg!({
    //~^ ERROR: expected type, found `direct_const_arg!()` constant
        fn foo() {}
        reuse foo::<>as bar;
        reuse bar;
        //~^ ERROR: the name `bar` is defined multiple times
    })
{
}

fn main() {}
