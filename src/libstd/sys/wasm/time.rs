// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime;

pub const UNIX_EPOCH: SystemTime = SystemTime;

impl Instant {
    pub fn now() -> Instant {
        panic!("not supported on web assembly");
    }

    pub fn sub_instant(&self, _other: &Instant) -> Duration {
        panic!("can't sub yet");
    }

    pub fn add_duration(&self, _other: &Duration) -> Instant {
        panic!("can't add yet");
    }

    pub fn sub_duration(&self, _other: &Duration) -> Instant {
        panic!("can't sub yet");
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        panic!("not supported on web assembly");
    }

    pub fn sub_time(&self, _other: &SystemTime)
                    -> Result<Duration, Duration> {
        panic!()
    }

    pub fn add_duration(&self, _other: &Duration) -> SystemTime {
        panic!()
    }

    pub fn sub_duration(&self, _other: &Duration) -> SystemTime {
        panic!()
    }
}

impl fmt::Debug for SystemTime {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        panic!()
    }
}
