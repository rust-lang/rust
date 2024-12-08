//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn foo(&self) {}
    fn bar(&self) {}
}

impl Trait for u8 {}

struct S(u8);

mod to_import {
    pub fn check(arg: &u8) -> &u8 { arg }
}

impl Trait for S {
    reuse Trait::* {
        use to_import::check;

        let _arr = Some(self.0).map(|x| [x * 2; 3]);
        check(&self.0)
    }
}

fn main() {
    let s = S(0);
    s.foo();
    s.bar();
}
