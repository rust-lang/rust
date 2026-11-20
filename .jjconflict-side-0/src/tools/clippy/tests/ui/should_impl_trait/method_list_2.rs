//@revisions: edition2015 edition2021
//@[edition2015] edition:2015
//@[edition2021] edition:2021
#![warn(clippy::should_implement_trait)]

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
        //~[edition2021]^ should_implement_trait

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
