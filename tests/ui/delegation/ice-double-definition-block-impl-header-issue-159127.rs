#![feature(min_generic_const_args, fn_delegation)]

impl
    core::direct_const_arg!({
    //~^ ERROR: expected type, found `direct_const_arg!()` constant
    //~| ERROR: complex const arguments must be placed inside of a `const` block
        fn foo() {}
        reuse foo as bar;
        reuse bar as baz;
    })
{
}

fn main() {}
