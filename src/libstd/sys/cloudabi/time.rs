// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mem;
use sys::cloudabi::abi;
use time::Duration;

const NSEC_PER_SEC: abi::timestamp = 1_000_000_000;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    t: abi::timestamp,
}

pub fn dur2intervals(dur: &Duration) -> abi::timestamp {
    dur.as_secs()
        .checked_mul(NSEC_PER_SEC)
        .and_then(|nanos| nanos.checked_add(dur.subsec_nanos() as abi::timestamp))
        .expect("overflow converting duration to nanoseconds")
}

impl Instant {
    pub fn now() -> Instant {
        unsafe {
            let mut t = mem::uninitialized();
            let ret = abi::clock_time_get(abi::clockid::MONOTONIC, 0, &mut t);
            assert_eq!(ret, abi::errno::SUCCESS);
            Instant { t: t }
        }
    }

    pub fn sub_instant(&self, other: &Instant) -> Duration {
        let diff = self.t
            .checked_sub(other.t)
            .expect("second instant is later than self");
        Duration::new(diff / NSEC_PER_SEC, (diff % NSEC_PER_SEC) as u32)
    }

    pub fn add_duration(&self, other: &Duration) -> Instant {
        Instant {
            t: self.t
                .checked_add(dur2intervals(other))
                .expect("overflow when adding duration to instant"),
        }
    }

    pub fn sub_duration(&self, other: &Duration) -> Instant {
        Instant {
            t: self.t
                .checked_sub(dur2intervals(other))
                .expect("overflow when subtracting duration from instant"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime {
    t: abi::timestamp,
}

impl SystemTime {
    pub fn now() -> SystemTime {
        unsafe {
            let mut t = mem::uninitialized();
            let ret = abi::clock_time_get(abi::clockid::REALTIME, 0, &mut t);
            assert_eq!(ret, abi::errno::SUCCESS);
            SystemTime { t: t }
        }
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        if self.t >= other.t {
            let diff = self.t - other.t;
            Ok(Duration::new(
                diff / NSEC_PER_SEC,
                (diff % NSEC_PER_SEC) as u32,
            ))
        } else {
            let diff = other.t - self.t;
            Err(Duration::new(
                diff / NSEC_PER_SEC,
                (diff % NSEC_PER_SEC) as u32,
            ))
        }
    }

    pub fn add_duration(&self, other: &Duration) -> SystemTime {
        SystemTime {
            t: self.t
                .checked_add(dur2intervals(other))
                .expect("overflow when adding duration to instant"),
        }
    }

    pub fn sub_duration(&self, other: &Duration) -> SystemTime {
        SystemTime {
            t: self.t
                .checked_sub(dur2intervals(other))
                .expect("overflow when subtracting duration from instant"),
        }
    }
}

pub const UNIX_EPOCH: SystemTime = SystemTime { t: 0 };
