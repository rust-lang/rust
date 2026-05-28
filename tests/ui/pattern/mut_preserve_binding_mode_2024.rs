//@ run-pass
//@ edition: 2024
#![feature(mut_ref, ref_pat_eat_one_layer_2024)]
#![allow(incomplete_features, unused)]

struct Foo(u8);

fn main() {
    let Foo(mut a) = &Foo(0);
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    a = &mut 42;
}
