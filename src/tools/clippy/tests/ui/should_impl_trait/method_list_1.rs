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
    // trait method list part 1, should lint all
    // *****************************************
    pub fn add(self, other: T) -> T {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn as_mut(&mut self) -> &mut T {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn as_ref(&self) -> &T {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn bitand(self, rhs: T) -> T {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn bitor(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn bitxor(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn borrow(&self) -> &str {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn borrow_mut(&mut self) -> &mut str {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn clone(&self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn cmp(&self, other: &Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn default() -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn deref(&self) -> &Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn deref_mut(&mut self) -> &mut Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn div(self, rhs: Self) -> Self {
        //~^ should_implement_trait

        unimplemented!()
    }

    pub fn drop(&mut self) {
        //~^ should_implement_trait

        unimplemented!()
    }
    // **********
    // part 1 end
    // **********
}
