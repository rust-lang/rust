//! Tests that multiple derefs in a projection does not cause an ICE
//! when checking const precise drops.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/147733>

#![feature(const_precise_live_drops)]
struct Foo(u32);
impl Foo {
    const fn get(self: Box<&Self>, f: &u32) -> u32 {
        //~^ ERROR destructor of `Box<&Foo>` cannot be evaluated at compile-time
        self.0
    }
}

fn main() {}
