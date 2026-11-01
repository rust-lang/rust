#![feature(fn_delegation)]
//~^ WARN the feature `fn_delegation` is incomplete

//@ pp-exact

trait Trait {
    fn bar(&self, x: i32) -> i32 { x }
}

struct F;
impl Trait for F {}

struct S(F);
impl Trait for S {
    reuse Trait::bar { &self.0 }
}

mod to_reuse {
    pub fn foo() {}
}

#[inline]
pub reuse to_reuse::foo;

fn main() {}
