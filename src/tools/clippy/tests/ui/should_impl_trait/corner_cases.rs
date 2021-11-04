#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::unused_self,
    clippy::needless_lifetimes,
    clippy::missing_safety_doc,
    clippy::wrong_self_convention,
    clippy::missing_panics_doc
)]

use std::ops::Mul;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

fn main() {}

pub struct T1;
impl T1 {
    // corner cases: should not lint

    // no error, not public interface
    pub(crate) fn drop(&mut self) {}

    // no error, private function
    fn neg(self) -> Self {
        self
    }

    // no error, private function
    fn eq(&self, other: Self) -> bool {
        true
    }

    // No error; self is a ref.
    fn sub(&self, other: Self) -> &Self {
        self
    }

    // No error; different number of arguments.
    fn div(self) -> Self {
        self
    }

    // No error; wrong return type.
    fn rem(self, other: Self) {}

    // Fine
    fn into_u32(self) -> u32 {
        0
    }

    fn into_u16(&self) -> u16 {
        0
    }

    fn to_something(self) -> u32 {
        0
    }

    fn new(self) -> Self {
        unimplemented!();
    }

    pub fn next<'b>(&'b mut self) -> Option<&'b mut T1> {
        unimplemented!();
    }
}

pub struct T2;
impl T2 {
    // Shouldn't trigger lint as it is unsafe.
    pub unsafe fn add(self, rhs: Self) -> Self {
        self
    }

    // Should not trigger lint since this is an async function.
    pub async fn next(&mut self) -> Option<Self> {
        None
    }
}
