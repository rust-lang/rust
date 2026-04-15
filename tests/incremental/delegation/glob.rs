//@ run-pass
//@ revisions:rpass1 rpass2

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
}

impl Trait for Z {
    reuse <u8 as Trait>::* { &self.0 }
}

fn main() {
    let s = S(2);
    s.foo();
    s.bar();

    let z = Z(3);
    z.foo();
    z.bar();
}
