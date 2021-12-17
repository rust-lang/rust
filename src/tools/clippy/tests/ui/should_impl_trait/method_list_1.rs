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
    // trait method list part 1, should lint all
    // *****************************************
    pub fn add(self, other: T) -> T {
        unimplemented!()
    }

    pub fn as_mut(&mut self) -> &mut T {
        unimplemented!()
    }

    pub fn as_ref(&self) -> &T {
        unimplemented!()
    }

    pub fn bitand(self, rhs: T) -> T {
        unimplemented!()
    }

    pub fn bitor(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn bitxor(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn borrow(&self) -> &str {
        unimplemented!()
    }

    pub fn borrow_mut(&mut self) -> &mut str {
        unimplemented!()
    }

    pub fn clone(&self) -> Self {
        unimplemented!()
    }

    pub fn cmp(&self, other: &Self) -> Self {
        unimplemented!()
    }

    pub fn default() -> Self {
        unimplemented!()
    }

    pub fn deref(&self) -> &Self {
        unimplemented!()
    }

    pub fn deref_mut(&mut self) -> &mut Self {
        unimplemented!()
    }

    pub fn div(self, rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn drop(&mut self) {
        unimplemented!()
    }
    // **********
    // part 1 end
    // **********
}
