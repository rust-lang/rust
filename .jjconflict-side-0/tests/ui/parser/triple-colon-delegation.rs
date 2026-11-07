//@ run-rustfix

#![feature(fn_delegation)]
#![allow(incomplete_features, unused)]

trait Trait {
    fn foo(&self) {}
}

struct F;
impl Trait for F {}

pub mod to_reuse {
    pub fn bar() {}
}

mod fn_to_other {
    use super::*;

    reuse Trait:::foo; //~ ERROR path separator must be a double colon
    reuse to_reuse:::bar; //~ ERROR path separator must be a double colon
}

impl Trait for u8 {}

struct S(u8);

mod to_import {
    pub fn check(arg: &u8) -> &u8 { arg }
}

impl Trait for S {
    reuse Trait:::* { //~ ERROR path separator must be a double colon
        use to_import::check;

        let _arr = Some(self.0).map(|x| [x * 2; 3]);
        check(&self.0)
    }
}

fn main() {
    let s = S(0);
    s.foo();
}
