#![feature(fn_delegation)]

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
    //~| ERROR: this function takes 1 argument but 0 arguments were supplied
    //~| ERROR: mismatched types
}

fn main() {}
