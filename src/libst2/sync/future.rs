// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A type representing values that may be computed concurrently and
 * operations for working with them.
 *
 * # Example
 *
 * ```rust
 * use std::sync::Future;
 * # fn fib(n: uint) -> uint {42};
 * # fn make_a_sandwich() {};
 * let mut delayed_fib = Future::spawn(proc() { fib(5000) });
 * make_a_sandwich();
 * println!("fib(5000) = {}", delayed_fib.get())
 * ```
 */

#![allow(missing_docs)]

use core::prelude::*;
use core::mem::replace;

use self::FutureState::*;
use comm::{Receiver, channel};
use task::spawn;

/// A type encapsulating the result of a computation which may not be complete
pub struct Future<A> {
    state: FutureState<A>,
}

enum FutureState<A> {
    Pending(proc():Send -> A),
    Evaluating,
    Forced(A)
}

/// Methods on the `future` type
impl<A:Clone> Future<A> {
    pub fn get(&mut self) -> A { unimplemented!() }
}

impl<A> Future<A> {
    /// Gets the value from this future, forcing evaluation.
    pub fn unwrap(mut self) -> A { unimplemented!() }

    pub fn get_ref<'a>(&'a mut self) -> &'a A { unimplemented!() }

    pub fn from_value(val: A) -> Future<A> { unimplemented!() }

    pub fn from_fn(f: proc():Send -> A) -> Future<A> { unimplemented!() }
}

impl<A:Send> Future<A> {
    pub fn from_receiver(rx: Receiver<A>) -> Future<A> { unimplemented!() }

    pub fn spawn(blk: proc():Send -> A) -> Future<A> { unimplemented!() }
}
