//@ check-pass

#![deny(non_camel_case_types)]

pub enum Foo1 {
    #[allow(non_camel_case_types)]
    bar,
}

pub enum Foo2 {
    Bar,
}
#[allow(non_camel_case_types)]
use Foo2::Bar as bar;

fn main() {}
