//@ check-pass

#![deny(non_camel_case_types)]

pub enum Foo {
    #[allow(non_camel_case_types)]
    bar
}

fn main() {}
