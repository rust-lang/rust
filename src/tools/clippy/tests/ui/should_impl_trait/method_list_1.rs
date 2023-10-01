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
        //~^ ERROR: method `add` can be confused for the standard trait method `std::ops::Add:
        unimplemented!()
    }

    pub fn as_mut(&mut self) -> &mut T {
        //~^ ERROR: method `as_mut` can be confused for the standard trait method `std::conver
        unimplemented!()
    }

    pub fn as_ref(&self) -> &T {
        //~^ ERROR: method `as_ref` can be confused for the standard trait method `std::conver
        unimplemented!()
    }

    pub fn bitand(self, rhs: T) -> T {
        //~^ ERROR: method `bitand` can be confused for the standard trait method `std::ops::B
        unimplemented!()
    }

    pub fn bitor(self, rhs: Self) -> Self {
        //~^ ERROR: method `bitor` can be confused for the standard trait method `std::ops::Bi
        unimplemented!()
    }

    pub fn bitxor(self, rhs: Self) -> Self {
        //~^ ERROR: method `bitxor` can be confused for the standard trait method `std::ops::B
        unimplemented!()
    }

    pub fn borrow(&self) -> &str {
        //~^ ERROR: method `borrow` can be confused for the standard trait method `std::borrow
        unimplemented!()
    }

    pub fn borrow_mut(&mut self) -> &mut str {
        //~^ ERROR: method `borrow_mut` can be confused for the standard trait method `std::bo
        unimplemented!()
    }

    pub fn clone(&self) -> Self {
        //~^ ERROR: method `clone` can be confused for the standard trait method `std::clone::
        unimplemented!()
    }

    pub fn cmp(&self, other: &Self) -> Self {
        //~^ ERROR: method `cmp` can be confused for the standard trait method `std::cmp::Ord:
        unimplemented!()
    }

    pub fn default() -> Self {
        //~^ ERROR: method `default` can be confused for the standard trait method `std::defau
        unimplemented!()
    }

    pub fn deref(&self) -> &Self {
        //~^ ERROR: method `deref` can be confused for the standard trait method `std::ops::De
        unimplemented!()
    }

    pub fn deref_mut(&mut self) -> &mut Self {
        //~^ ERROR: method `deref_mut` can be confused for the standard trait method `std::ops
        unimplemented!()
    }

    pub fn div(self, rhs: Self) -> Self {
        //~^ ERROR: method `div` can be confused for the standard trait method `std::ops::Div:
        unimplemented!()
    }

    pub fn drop(&mut self) {
        //~^ ERROR: method `drop` can be confused for the standard trait method `std::ops::Dro
        unimplemented!()
    }
    // **********
    // part 1 end
    // **********
}
