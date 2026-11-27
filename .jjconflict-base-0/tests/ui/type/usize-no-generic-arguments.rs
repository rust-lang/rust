//! Sanity test that primitives cannot have const generics.

fn foo() {
    let x: usize<foo>; //~ ERROR const arguments are not allowed on builtin type `usize`
}

fn main() {}
