#![feature(min_generic_const_args)]
#![feature(fn_delegation)]

pub struct S<const N: usize>;

impl
    S<
        core::direct_const_arg!({
            fn foo() {}
            reuse foo::<> as bar;
            reuse bar;
            //~^ ERROR: the name `bar` is defined multiple times
        }),
    >
{
}

fn main() {}
