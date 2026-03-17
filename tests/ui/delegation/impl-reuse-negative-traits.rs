#![allow(incomplete_features)]
#![feature(fn_delegation)]
#![feature(negative_impls)]

trait Trait {
    fn foo(&self);
}

struct S;
impl Trait for S {
    fn foo(&self) {}
}

struct F(S);

reuse impl !Trait for F { &self.0 }
//~^ ERROR negative impls cannot have any items [E0749]

fn main() {}
