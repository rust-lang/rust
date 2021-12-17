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
        unimplemented!()
    }

    pub fn from_iter<T>(iter: T) -> Self {
        unimplemented!()
    }

    pub fn from_str(s: &str) -> Result<Self, Self> {
        unimplemented!()
    }

    pub fn hash(&self, state: &mut T) {
        unimplemented!()
    }

    pub fn index(&self, index: usize) -> &Self {
        unimplemented!()
    }

    pub fn index_mut(&mut self, index: usize) -> &mut Self {
        unimplemented!()
    }

    pub fn into_iter(self) -> Self {
        unimplemented!()
    }

    pub fn mul(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn neg(self) -> Self {
        unimplemented!()
    }

    pub fn next(&mut self) -> Option<Self> {
        unimplemented!()
    }

    pub fn not(self) -> Self {
        unimplemented!()
    }

    pub fn rem(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn shl(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn shr(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn sub(self, rhs: Self) -> Self {
        unimplemented!()
    }
    // **********
    // part 2 end
    // **********
}
