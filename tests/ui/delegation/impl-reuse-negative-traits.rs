#![allow(incomplete_features)]
#![feature(fn_delegation)]
#![feature(negative_impls)]

trait Trait {
    fn foo(&self);
    //~^ ERROR negative impls cannot have any items [E0749]
}

struct S;
impl Trait for S {
    fn foo(&self) {}
}

struct F(S);

reuse impl !Trait for F { &self.0 }

fn main() {}
