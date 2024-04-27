#![warn(clippy::all, clippy::pedantic)]
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
        //~^ ERROR: method `eq` can be confused for the standard trait method `std::cmp::Parti
        unimplemented!()
    }

    pub fn from_iter<T>(iter: T) -> Self {
        //~^ ERROR: method `from_iter` can be confused for the standard trait method `std::ite
        unimplemented!()
    }

    pub fn from_str(s: &str) -> Result<Self, Self> {
        //~^ ERROR: method `from_str` can be confused for the standard trait method `std::str:
        unimplemented!()
    }

    pub fn hash(&self, state: &mut T) {
        //~^ ERROR: method `hash` can be confused for the standard trait method `std::hash::Ha
        unimplemented!()
    }

    pub fn index(&self, index: usize) -> &Self {
        //~^ ERROR: method `index` can be confused for the standard trait method `std::ops::In
        unimplemented!()
    }

    pub fn index_mut(&mut self, index: usize) -> &mut Self {
        //~^ ERROR: method `index_mut` can be confused for the standard trait method `std::ops
        unimplemented!()
    }

    pub fn into_iter(self) -> Self {
        //~^ ERROR: method `into_iter` can be confused for the standard trait method `std::ite
        unimplemented!()
    }

    pub fn mul(self, rhs: Self) -> Self {
        //~^ ERROR: method `mul` can be confused for the standard trait method `std::ops::Mul:
        unimplemented!()
    }

    pub fn neg(self) -> Self {
        //~^ ERROR: method `neg` can be confused for the standard trait method `std::ops::Neg:
        unimplemented!()
    }

    pub fn next(&mut self) -> Option<Self> {
        //~^ ERROR: method `next` can be confused for the standard trait method `std::iter::It
        unimplemented!()
    }

    pub fn not(self) -> Self {
        //~^ ERROR: method `not` can be confused for the standard trait method `std::ops::Not:
        unimplemented!()
    }

    pub fn rem(self, rhs: Self) -> Self {
        //~^ ERROR: method `rem` can be confused for the standard trait method `std::ops::Rem:
        unimplemented!()
    }

    pub fn shl(self, rhs: Self) -> Self {
        //~^ ERROR: method `shl` can be confused for the standard trait method `std::ops::Shl:
        unimplemented!()
    }

    pub fn shr(self, rhs: Self) -> Self {
        //~^ ERROR: method `shr` can be confused for the standard trait method `std::ops::Shr:
        unimplemented!()
    }

    pub fn sub(self, rhs: Self) -> Self {
        //~^ ERROR: method `sub` can be confused for the standard trait method `std::ops::Sub:
        unimplemented!()
    }
    // **********
    // part 2 end
    // **********
}
