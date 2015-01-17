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
        self.t as u64 * 1_000_000_000 / frequency() as u64
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
        Duration::microseconds(diff * 1_000_000 / frequency() as i64)
    }
}
