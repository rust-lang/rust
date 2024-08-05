//@ run-pass
#![allow(non_camel_case_types)]

macro_rules! define_vec {
    () => (
        mod foo {
            #[derive(PartialEq)]
            pub struct bar; //~ WARN struct `bar` is never constructed
        }
    )
}

define_vec![];

pub fn main() {}
