//@ run-pass
//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![feature(mut_preserve_binding_mode_2024)]
#![allow(incomplete_features, unused)]

struct Foo(u8);

fn main() {
    let Foo(mut a) = &Foo(0);
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    a = &mut 42;
}
