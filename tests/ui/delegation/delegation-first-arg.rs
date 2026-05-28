#![feature(fn_delegation)]

trait Trait: Sized {
    fn value(self) {}
    fn r#ref(&self) {}
    fn mut_ref(&mut self) {}

    fn static_empty() {}
    fn static_one_param(x: usize) {}
}

struct S;
impl Trait for S {}

struct F(S);
// In glob delegations silently remove first arg if no params or generate default
// first arg (`arg0`) if it is a static function.
reuse impl Trait for F { self.0 }
//~^ ERROR: type annotations needed
//~| ERROR: type annotations needed

struct F1(S);
impl F1 {
    reuse Trait::{value, r#ref, mut_ref} { self.0 }

    // Error is reported as user has explicitly specified block when no params.
    reuse <S as Trait>::static_empty { self.0 }
    //~^ ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied

    reuse <S as Trait>::static_one_param { self.0 }
    //~^ ERROR: `usize` is a primitive type and therefore doesn't have fields
}

struct F2(S);
impl F2 {
    // In list delegations silently remove first arg if it is not a method.
    reuse <S as Trait>::{value, r#ref, mut_ref, static_empty, static_one_param} { self.0 }
}

mod trait_to_reuse {
    use super::Trait;

    pub fn value(_: impl Trait) {}
    pub fn r#ref(_: &impl Trait) {}
    pub fn mut_ref(_: &mut impl Trait) {}

    pub fn static_empty() {}
    pub fn static_one_param(x: usize) {}
}

struct F3(S);
impl Trait for F3 {
    reuse trait_to_reuse::{value, r#ref, mut_ref, static_empty, static_one_param} { self.0 }
    //~^ ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct F4(S);
impl F4 {
    reuse trait_to_reuse::{value, r#ref, mut_ref, static_empty, static_one_param} { self.0 }
    //~^ ERROR: no field `0` on type `impl Trait`
    //~| ERROR: no field `0` on type `&impl Trait`
    //~| ERROR: no field `0` on type `&mut impl Trait`
    //~| ERROR: `usize` is a primitive type and therefore doesn't have fields
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
    //~| ERROR: delegation's target expression is specified for function with no params
}

mod to_reuse {
    pub fn empty() {}
    pub fn one_param(x: usize) {}
}

// Error is reported as user has explicitly specified block when no params.
reuse to_reuse::empty { self + 1 }
//~^ ERROR: delegation's target expression is specified for function with no params
//~| ERROR: this function takes 0 arguments but 1 argument was supplied

reuse to_reuse::one_param { self + 1 }

reuse to_reuse::{empty as empty1, one_param as one_param1} { self + 1 }
//~^ ERROR: this function takes 0 arguments but 1 argument was supplied
//~| ERROR: delegation's target expression is specified for function with no params

fn main() {}
