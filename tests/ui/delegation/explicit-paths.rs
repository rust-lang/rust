#![feature(fn_delegation)]
//~^ WARN the feature `fn_delegation` is incomplete

trait Trait {
    fn bar(&self) -> i32 { 42 }
}

struct F;
impl Trait for F {}

struct S(F);

impl Trait for S {
    reuse <F as Trait>::bar;
    //~^ ERROR mismatched types
}

struct S2(F);

impl Trait for S2 {
    reuse <S2 as Trait>::bar { &self.0 }
    //~^ ERROR mismatched types
}

fn main() {}
