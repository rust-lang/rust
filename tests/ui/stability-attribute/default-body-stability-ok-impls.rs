//@ check-pass
//@ aux-build:default_body.rs
#![crate_type = "lib"]

extern crate default_body;

use default_body::{Equal, JustTrait};

struct Type;

impl JustTrait for Type {
    const CONSTANT: usize = 1;

    fn fun() {}

    fn fun2() {}
}

impl Equal for Type {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}
