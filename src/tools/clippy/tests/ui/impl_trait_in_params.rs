#![allow(unused)]
#![warn(clippy::impl_trait_in_params)]

//@no-rustfix
pub trait Trait {}
pub trait AnotherTrait<T> {}

// Should warn
pub fn a(_: impl Trait) {}
//~^ ERROR: `impl Trait` used as a function parameter
pub fn c<C: Trait>(_: C, _: impl Trait) {}
//~^ ERROR: `impl Trait` used as a function parameter

// Shouldn't warn

pub fn b<B: Trait>(_: B) {}
fn e<T: AnotherTrait<u32>>(_: T) {}
fn d(_: impl AnotherTrait<u32>) {}

//------ IMPLS

pub trait Public {
    // See test in ui-toml for a case where avoid-breaking-exported-api is set to false
    fn t(_: impl Trait);
    fn tt<T: Trait>(_: T) {}
}

trait Private {
    // This shouldn't lint
    fn t(_: impl Trait);
    fn tt<T: Trait>(_: T) {}
}

struct S;
impl S {
    pub fn h(_: impl Trait) {} //~ ERROR: `impl Trait` used as a function parameter
    fn i(_: impl Trait) {}
    pub fn j<J: Trait>(_: J) {}
    pub fn k<K: AnotherTrait<u32>>(_: K, _: impl AnotherTrait<u32>) {} //~ ERROR: `impl Trait` used as a function parameter
}

// Trying with traits
impl Public for S {
    fn t(_: impl Trait) {}
}

fn main() {}
