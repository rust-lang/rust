//@ edition: 2021
//@ compile-flags: -Zunstable-options
#![feature(ref_pat_eat_one_layer_2024)]
#![allow(incomplete_features)]

struct Foo(u8);

fn main() {
    let Foo(mut a) = &Foo(0);
    a = &42;
    //~^ ERROR: mismatched types

    let Foo(mut a) = &mut Foo(0);
    a = &mut 42;
    //~^ ERROR: mismatched types
}
