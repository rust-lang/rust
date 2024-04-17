//@ edition: 2021
#![feature(mut_preserve_binding_mode_2024)]
#![allow(incomplete_features, unused)]
#![forbid(dereferencing_mut_binding)]

struct Foo(u8);

fn main() {
    let Foo(mut a) = &Foo(0);
    //~^ ERROR: dereferencing `mut` binding
    a = 42;

    let Foo(mut a) = &mut Foo(0);
    //~^ ERROR: dereferencing `mut` binding
    a = 42;
}
