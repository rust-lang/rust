//@ check-pass
#![allow(non_camel_case_types)]
#![allow(dead_code)]

macro_rules! define_vec {
    () => (
        mod foo {
            #[derive(PartialEq)]
            pub struct bar;
        }
    )
}

define_vec![];

pub fn main() {}
