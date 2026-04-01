#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn foo(&self) -> u32 { 0 }
}

struct F;
struct S;

mod to_reuse {
    use crate::S;

    pub fn foo(_: &S) -> u32 { 0 }
}

impl Trait for S {
    reuse to_reuse::foo { self }
    reuse Trait::foo;
    //~^ ERROR  duplicate definitions with name `foo`
}

fn main() {}
