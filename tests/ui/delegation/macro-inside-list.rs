//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn foo(&self) -> u8 { 0 }
    fn bar(&self) -> u8 { 1 }
}

impl Trait for u8 {}

struct S(u8);

// Macro expansion works inside delegation items.
macro_rules! u8 { () => { u8 } }
macro_rules! self_0 { ($self:ident) => { &$self.0 } }
impl Trait for S {
    reuse <u8!() as Trait>::{foo, bar} { self_0!(self) }
}

fn main() {
    let s = S(2);
    s.foo();
    s.bar();
}
