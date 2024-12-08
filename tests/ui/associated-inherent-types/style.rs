#![feature(inherent_associated_types)]
#![allow(incomplete_features, dead_code)]
#![deny(non_camel_case_types)]

struct S;

impl S {
    type typ = ();
    //~^ ERROR associated type `typ` should have an upper camel case name
}

fn main() {}
