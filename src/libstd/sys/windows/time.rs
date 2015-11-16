// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ops::Sub;
use sync::Once;
use sys::c;
use time::Duration;

const NANOS_PER_SEC: u64 = 1_000_000_000;

pub struct SteadyTime {
    t: c::LARGE_INTEGER,
}

impl SteadyTime {
    pub fn now() -> SteadyTime {
        let mut t = SteadyTime { t: 0 };
        unsafe { c::QueryPerformanceCounter(&mut t.t); }
        t
    }
}

fn frequency() -> c::LARGE_INTEGER {
    static mut FREQUENCY: c::LARGE_INTEGER = 0;
    static ONCE: Once = Once::new();

    unsafe {
        ONCE.call_once(|| {
            c::QueryPerformanceFrequency(&mut FREQUENCY);
        });
        FREQUENCY
    }
}

#[unstable(feature = "libstd_sys_internals", issue = "0")]
impl<'a> Sub for &'a SteadyTime {
    type Output = Duration;

    fn sub(self, other: &SteadyTime) -> Duration {
        let diff = self.t as u64 - other.t as u64;
        let nanos = mul_div_u64(diff, NANOS_PER_SEC, frequency() as u64);
        Duration::new(nanos / NANOS_PER_SEC, (nanos % NANOS_PER_SEC) as u32)
    }
}

// Computes (value*numer)/denom without overflow, as long as both
// (numer*denom) and the overall result fit into i64 (which is the case
// for our time conversions).
fn mul_div_u64(value: u64, numer: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numer)/denom and simplify.
    // r < denom, so (denom*numer) is the upper bound of (r*numer)
    q * numer + r * numer / denom
}

#[test]
fn test_muldiv() {
    assert_eq!(mul_div_u64( 1_000_000_000_001, 1_000_000_000, 1_000_000),
               1_000_000_000_001_000);
}
