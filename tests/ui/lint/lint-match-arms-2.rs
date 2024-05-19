#![feature(if_let_guard)]
#![allow(unused, non_snake_case)]

enum E {
    A,
}

#[allow(bindings_with_variant_name, irrefutable_let_patterns)]
fn foo() {
    match E::A {
        #[deny(bindings_with_variant_name)]
        A => {}
    //~^ ERROR pattern binding `A` is named the same as one of the variants of the type `E`
    }

    match &E::A {
        #[deny(irrefutable_let_patterns)]
        a if let b = a => {}
    //~^ ERROR irrefutable `if let` guard pattern
        _ => {}
    }
}

fn main() { }
