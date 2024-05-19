#![deny(non_upper_case_globals)]
#![allow(dead_code)]

struct Foo;

impl Foo {
    const not_upper: bool = true;
}
//~^^ ERROR associated constant `not_upper` should have an upper case name

fn main() {}
