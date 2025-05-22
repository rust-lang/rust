#![allow(non_camel_case_types)]
#![deny(dead_code)]

macro_rules! define_vec {
    () => (
        mod foo {
            #[derive(PartialEq)]
            pub struct bar; //~ ERROR struct `bar` is never constructed
        }
    )
}

define_vec![];

pub fn main() {}
