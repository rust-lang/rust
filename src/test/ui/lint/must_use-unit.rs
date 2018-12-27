#![feature(never_type)]

#![deny(unused_must_use)]

#[must_use]
fn foo() {}

#[must_use]
fn bar() -> ! {
    unimplemented!()
}

fn main() {
    foo(); //~ unused return value of `foo`

    bar(); //~ unused return value of `bar`
}
