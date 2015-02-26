// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use libc;
use ops::Sub;
use time::Duration;
use sync::{Once, ONCE_INIT};

const NANOS_PER_SEC: i64 = 1_000_000_000;

pub struct SteadyTime {
    t: libc::LARGE_INTEGER,
}

impl SteadyTime {
    pub fn now() -> SteadyTime {
        let mut t = SteadyTime { t: 0 };
        unsafe { libc::QueryPerformanceCounter(&mut t.t); }
        t
    }

    pub fn ns(&self) -> u64 {
        mul_div_i64(self.t as i64, NANOS_PER_SEC, frequency() as i64) as u64
    }
}

fn frequency() -> libc::LARGE_INTEGER {
    static mut FREQUENCY: libc::LARGE_INTEGER = 0;
    static ONCE: Once = ONCE_INIT;

    unsafe {
        ONCE.call_once(|| {
            libc::QueryPerformanceFrequency(&mut FREQUENCY);
        });
        FREQUENCY
    }
}

impl<'a> Sub for &'a SteadyTime {
    type Output = Duration;

    fn sub(self, other: &SteadyTime) -> Duration {
        let diff = self.t as i64 - other.t as i64;
        Duration::nanoseconds(mul_div_i64(diff, NANOS_PER_SEC, frequency() as i64))
    }
}

// Computes (value*numer)/denom without overflow, as long as both
// (numer*denom) and the overall result fit into i64 (which is the case
// for our time conversions).
fn mul_div_i64(value: i64, numer: i64, denom: i64) -> i64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numer)/denom and simplify.
    // r < denom, so (denom*numer) is the upper bound of (r*numer)
    q * numer + r * numer / denom
}

#[test]
fn test_muldiv() {
    assert_eq!(mul_div_i64( 1_000_000_000_001, 1_000_000_000, 1_000_000),  1_000_000_000_001_000);
    assert_eq!(mul_div_i64(-1_000_000_000_001, 1_000_000_000, 1_000_000), -1_000_000_000_001_000);
    assert_eq!(mul_div_i64(-1_000_000_000_001,-1_000_000_000, 1_000_000),  1_000_000_000_001_000);
    assert_eq!(mul_div_i64( 1_000_000_000_001, 1_000_000_000,-1_000_000), -1_000_000_000_001_000);
    assert_eq!(mul_div_i64( 1_000_000_000_001,-1_000_000_000,-1_000_000),  1_000_000_000_001_000);
}
