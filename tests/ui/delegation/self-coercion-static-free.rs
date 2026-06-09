// Test that we do not adjust first argument type to first parameter type
// in delegations to static associated functions (we only do it for methods).
// Also test that we do not adjust first arg. type in delegations to free function.

#![feature(fn_delegation)]

pub trait Trait: Sized {
    fn static_self() -> F { F }

    fn static_value(_: Self) -> i32 { 1 }
    fn static_mut_ref(_: &mut Self) -> i32 { 2 }
    fn static_ref(_: &Self) -> i32 { 3 }
}

#[derive(Default)]
struct F;
impl Trait for F {}

struct S(F);

impl Trait for S {
    reuse <F as Trait>::{static_value, static_mut_ref, static_ref} {
        //~^ ERROR: mismatched types
        //~| ERROR: mismatched types
        //~| ERROR: mismatched types
        let _ = self;
        S::static_self()
    }
}

struct S1(Box<Box<Box<Box<Box<Box<F>>>>>>);

impl Trait for S1 {
    reuse <F as Trait>::{static_value, static_mut_ref, static_ref} {
        //~^ ERROR: mismatched types
        //~| ERROR: mismatched types
        //~| ERROR: mismatched types
        let _ = self;
        S1::static_self()
    }
}

mod to_reuse {
    use super::Trait;
    pub fn value(_: impl Trait) -> i32 { 1 }
    pub fn mut_ref(_: &mut impl Trait) -> i32 { 2 }
    pub fn r#ref(_: &impl Trait) -> i32 { 3 }
}

reuse to_reuse::{value, mut_ref, r#ref} { F }
//~^ ERROR: mismatched types
//~| ERROR: mismatched types

fn main() {}
