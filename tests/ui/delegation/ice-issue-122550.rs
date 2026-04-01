#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn description(&self) -> &str {}
    //~^ ERROR mismatched types
}

struct F;
struct S(F);

impl S {
    reuse <S as Trait>::description { &self.0 }
    //~^ ERROR mismatched types
    //~| ERROR the trait bound `S: Trait` is not satisfied
}

fn main() {}
