// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporal quantification.

#![stable(feature = "time", since = "1.3.0")]

pub use core::time::Duration;

use sys::time::prelude::*;

/// Runs a closure, returning the duration of time it took to run the
/// closure.
#[unstable(feature = "time_span",
           reason = "unsure if this is the right API or whether it should \
                     wait for a more general \"moment in time\" \
                     abstraction",
           issue = "27799")]
pub fn span<F>(f: F) -> Duration where F: FnOnce() {
    let start = SteadyTime::now().unwrap();
    f();
    SteadyTime::now().unwrap().delta(&start)
}
