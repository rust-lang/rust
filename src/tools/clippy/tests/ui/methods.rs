// aux-build:option_helpers.rs

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::disallowed_names,
    clippy::default_trait_access,
    clippy::let_underscore_untyped,
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
    clippy::unused_async,
    clippy::unused_self,
    unused
)]

#[macro_use]
extern crate option_helpers;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ops::Mul;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

use option_helpers::{IteratorFalsePositives, IteratorMethodFalsePositives};

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

struct T;

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

    // Multi-line case.
    let _ = v.iter().filter(|&x| {
                                *x < 0
                            }
                   ).next();

    // Check that we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.filter().next();

    let foo = IteratorMethodFalsePositives {};
    let _ = foo.filter(42).next();
}

fn main() {
    filter_next();
}
