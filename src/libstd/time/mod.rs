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

#![unstable(feature = "std_misc")]

use sys::time::SteadyTime;

pub use self::duration::Duration;

pub mod duration;

/// Returns the current value of a high-resolution performance counter
/// in nanoseconds since an unspecified epoch.
// NB: this is intentionally not public, this is not ready to stabilize its api.
fn precise_time_ns() -> u64 {
    SteadyTime::now().ns()
}
