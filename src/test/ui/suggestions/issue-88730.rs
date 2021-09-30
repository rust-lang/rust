#![allow(unused, nonstandard_style)]
#![deny(bindings_with_variant_name)]

// If an enum has two different variants,
// then it cannot be matched upon in a function argument.
// It still gets a warning, but no suggestions.
enum Foo {
    C,
    D,
}

fn foo(C: Foo) {} //~ERROR

fn main() {
    let C = Foo::D; //~ERROR
}
