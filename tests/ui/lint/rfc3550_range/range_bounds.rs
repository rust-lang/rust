#![crate_type="lib"]

#![deny(range_bounds)]
#![allow(dead_code)]

use std::ops::RangeBounds;

// Private, no error.
fn p<R: RangeBounds<usize>>(range: R) {}

pub fn f_std<R: RangeBounds<usize>>(range: R) {} //~ ERROR usage of `RangeBounds` trait bound in public function
pub fn f_core<R: core::ops::RangeBounds<usize>>(range: R) {} //~ ERROR usage of `RangeBounds` trait bound in public function
pub fn f_impl_trait(range: impl RangeBounds<usize>) {} //~ ERROR usage of `RangeBounds` trait bound in public function
pub fn f_generic<B, R: RangeBounds<B>>(range: R) {} //~ ERROR usage of `RangeBounds` trait bound in public function
pub fn f_where<R>(range: R) where R: RangeBounds<usize> {} //~ ERROR usage of `RangeBounds` trait bound in public function

struct Priv<R>(R);
impl<R: RangeBounds<usize>> Priv<R> {
    pub fn new(range: R) -> Self {
        Self(range)
    }
}

pub struct Pub<R>(R);
impl<R: RangeBounds<usize>> Pub<R> { //~ ERROR usage of `RangeBounds` trait bound in public struct impl
    pub fn new(range: R) -> Self {
        Self(range)
    }
}

pub struct Thing;
impl Thing {
    fn p(range: impl RangeBounds<usize>) {}
    pub fn bar(range: impl RangeBounds<usize>) {} //~ ERROR usage of `RangeBounds` trait bound in public function
}

pub struct Wrapper<R: RangeBounds<usize>>(R); //~ ERROR usage of `RangeBounds` trait bound in public struct definition

pub trait Action<R: RangeBounds<usize>> {} //~ ERROR usage of `RangeBounds` trait bound in public trait definition
pub trait Super: RangeBounds<usize> {} //~ ERROR usage of `RangeBounds` trait bound in public trait definition
