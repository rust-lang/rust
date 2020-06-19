// aux-build:option_helpers.rs
// edition:2018

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::blacklisted_name,
    clippy::default_trait_access,
    clippy::missing_docs_in_private_items,
    clippy::missing_safety_doc,
    clippy::non_ascii_literal,
    clippy::new_without_default,
    clippy::needless_pass_by_value,
    clippy::needless_lifetimes,
    clippy::print_stdout,
    clippy::must_use_candidate,
    clippy::use_self,
    clippy::useless_format,
    clippy::wrong_self_convention,
    clippy::unused_self,
    unused
)]

#[macro_use]
extern crate option_helpers;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::iter::FromIterator;
use std::ops::Mul;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

use option_helpers::IteratorFalsePositives;

pub struct T;

impl T {
    // *******************************************
    // complete trait method list, should lint all
    // *******************************************
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
    // *****************
    // complete list end
    // *****************
}

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

    pub fn next<'b>(&'b mut self) -> Option<&'b mut T> {
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

struct Lt<'a> {
    foo: &'a u32,
}

impl<'a> Lt<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    #[allow(clippy::needless_lifetimes)]
    pub fn new<'b>(s: &'b str) -> Lt<'b> {
        unimplemented!()
    }
}

struct Lt2<'a> {
    foo: &'a u32,
}

impl<'a> Lt2<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    pub fn new(s: &str) -> Lt2 {
        unimplemented!()
    }
}

struct Lt3<'a> {
    foo: &'a u32,
}

impl<'a> Lt3<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    pub fn new() -> Lt3<'static> {
        unimplemented!()
    }
}

#[derive(Clone, Copy)]
struct U;

impl U {
    fn new() -> Self {
        U
    }
    // Ok because `U` is `Copy`.
    fn to_something(self) -> u32 {
        0
    }
}

struct V<T> {
    _dummy: T,
}

impl<T> V<T> {
    fn new() -> Option<V<T>> {
        None
    }
}

struct AsyncNew;

impl AsyncNew {
    async fn new() -> Option<Self> {
        None
    }
}

struct BadNew;

impl BadNew {
    fn new() -> i32 {
        0
    }
}

impl Mul<T> for T {
    type Output = T;
    // No error, obviously.
    fn mul(self, other: T) -> T {
        self
    }
}

/// Checks implementation of `FILTER_NEXT` lint.
#[rustfmt::skip]
fn filter_next() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // Single-line case.
    let _ = v.iter().filter(|&x| *x < 0).next();

    // Multi-line case.
    let _ = v.iter().filter(|&x| {
                                *x < 0
                            }
                   ).next();

    // Check that hat we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.filter().next();
}

/// Checks implementation of `SEARCH_IS_SOME` lint.
#[rustfmt::skip]
fn search_is_some() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;

    // Check `find().is_some()`, single-line case.
    let _ = v.iter().find(|&x| *x < 0).is_some();
    let _ = (0..1).find(|x| **y == *x).is_some(); // one dereference less
    let _ = (0..1).find(|x| *x == 0).is_some();
    let _ = v.iter().find(|x| **x == 0).is_some();

    // Check `find().is_some()`, multi-line case.
    let _ = v.iter().find(|&x| {
                              *x < 0
                          }
                   ).is_some();

    // Check `position().is_some()`, single-line case.
    let _ = v.iter().position(|&x| x < 0).is_some();

    // Check `position().is_some()`, multi-line case.
    let _ = v.iter().position(|&x| {
                                  x < 0
                              }
                   ).is_some();

    // Check `rposition().is_some()`, single-line case.
    let _ = v.iter().rposition(|&x| x < 0).is_some();

    // Check `rposition().is_some()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
                                   x < 0
                               }
                   ).is_some();

    // Check that we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.find().is_some();
    let _ = foo.position().is_some();
    let _ = foo.rposition().is_some();
}

fn main() {
    filter_next();
    search_is_some();
}
