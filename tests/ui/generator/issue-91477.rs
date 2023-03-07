#![feature(generators)]

fn foo() -> impl Sized {
    yield 1; //~ ERROR E0627
}

fn main() {}
