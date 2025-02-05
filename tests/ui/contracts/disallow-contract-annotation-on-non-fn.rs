//! Checks for compilation errors related to adding contracts to non-function items.

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
#![allow(dead_code)]

#[core::contracts::requires(true)]
//~^ ERROR contract annotations can only be used on functions
struct Dummy(usize);

#[core::contracts::ensures(|v| v == 100)]
//~^ ERROR contract annotations can only be used on functions
const MAX_VAL: usize = 100;

// FIXME: Improve the error message here. The macro thinks this is a function.
#[core::contracts::ensures(|v| v == 100)]
//~^ ERROR contract annotations is only supported in functions with bodies
type NewDummy = fn(usize) -> Dummy;

#[core::contracts::ensures(|v| v == 100)]
//~^ ERROR contract annotations is only supported in functions with bodies
const NEW_DUMMY_FN : fn(usize) -> Dummy = || { Dummy(0) };

#[core::contracts::requires(true)]
//~^ ERROR contract annotations can only be used on functions
impl Dummy {

    // This should work
    #[core::contracts::ensures(|ret| ret.0 == v)]
    fn new(v: usize) -> Dummy {
        Dummy(v)
    }
}

#[core::contracts::ensures(|dummy| dummy.0 > 0)]
//~^ ERROR contract annotations can only be used on functions
impl From<usize> for Dummy {
    // This should work.
    #[core::contracts::ensures(|ret| ret.0 == v)]
    fn from(value: usize) -> Self {
        Dummy::new(value)
    }
}

/// You should not be able to annotate a trait either.
#[core::contracts::requires(true)]
//~^ ERROR contract annotations can only be used on functions
pub trait DummyBuilder {
    fn build() -> Dummy;
}

fn main() {
}
