//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn foo(&self) -> u8;
    fn bar(&self) -> u8;
}

impl Trait for u8 {
    fn foo(&self) -> u8 { 0 }
    fn bar(&self) -> u8 { 1 }
}

struct S(u8);
struct Z(u8);

impl Trait for S {
    reuse Trait::* { &self.0 }
    fn bar(&self) -> u8 { 2 }
}

impl Trait for Z {
    reuse Trait::* { &self.0 }
    reuse Trait::bar { &self.0 }
}

fn main() {
    let s = S(2);
    s.foo();
    s.bar();

    let z = Z(2);
    z.foo();
    z.bar();
}
