// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Routines are like closures except that they own their arguments and can
 * only run once.
 */

/// A routine that takes no arguments and returns nothing.
pub trait Runnable {
    /// The entry point for the routine.
    fn run(~self);
}

/// A convenience routine that does nothing.
pub struct NoOpRunnable;

impl Runnable for NoOpRunnable {
    fn run(~self) {}
}

