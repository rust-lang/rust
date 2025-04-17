#![allow(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::unused_self,
    clippy::needless_lifetimes,
    clippy::missing_safety_doc,
    clippy::wrong_self_convention,
    clippy::missing_panics_doc,
    clippy::return_self_not_must_use
)]
//@no-rustfix
use std::ops::Mul;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

fn main() {}
pub struct T;

impl T {
    // *****************************************
    // trait method list part 2, should lint all
    // *****************************************

    pub fn eq(&self, other: &Self) -> bool {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn from_iter<T>(iter: T) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn from_str(s: &str) -> Result<Self, Self> {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn hash(&self, state: &mut T) {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn index(&self, index: usize) -> &Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn index_mut(&mut self, index: usize) -> &mut Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn into_iter(self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn mul(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn neg(self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn next(&mut self) -> Option<Self> {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn not(self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn rem(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn shl(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn shr(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn sub(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }
    // **********
    // part 2 end
    // **********
}
