#![feature(min_generic_const_args, opaque_generic_const_args)]
#![expect(incomplete_features)]

fn foo<const N: usize>() {}
fn bar<const N: usize>() {
    foo::<const { N + 1 }>();
               //~^ ERROR: generic parameters in const blocks are only allowed as the direct value of a `type const`
}
fn main(){}
