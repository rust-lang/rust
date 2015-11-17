// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::inner::{Instant, SystemTime, UNIX_EPOCH};

const NSEC_PER_SEC: u64 = 1_000_000_000;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod inner {
    use cmp::Ordering;
    use fmt;
    use libc;
    use super::NSEC_PER_SEC;
    use sync::Once;
    use sys::cvt;
    use sys_common::mul_div_u64;
    use time::Duration;

    const USEC_PER_SEC: u64 = NSEC_PER_SEC / 1000;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct Instant {
        t: u64
    }

    #[derive(Copy, Clone)]
    pub struct SystemTime {
        t: libc::timeval,
    }

    pub const UNIX_EPOCH: SystemTime = SystemTime {
        t: libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
    };

    impl Instant {
        pub fn now() -> Instant {
            Instant { t: unsafe { libc::mach_absolute_time() } }
        }

        pub fn sub_instant(&self, other: &Instant) -> Duration {
            let info = info();
            let diff = self.t.checked_sub(other.t)
                           .expect("second instant is later than self");
            let nanos = mul_div_u64(diff, info.numer as u64, info.denom as u64);
            Duration::new(nanos / NSEC_PER_SEC, (nanos % NSEC_PER_SEC) as u32)
        }

        pub fn add_duration(&self, other: &Duration) -> Instant {
            Instant {
                t: self.t.checked_add(dur2intervals(other))
                       .expect("overflow when adding duration to instant"),
            }
        }

        pub fn sub_duration(&self, other: &Duration) -> Instant {
            Instant {
                t: self.t.checked_sub(dur2intervals(other))
                       .expect("overflow when adding duration to instant"),
            }
        }
    }

    impl SystemTime {
        pub fn now() -> SystemTime {
            let mut s = SystemTime {
                t: libc::timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                },
            };
            cvt(unsafe {
                libc::gettimeofday(&mut s.t, 0 as *mut _)
            }).unwrap();
            return s
        }

        pub fn sub_time(&self, other: &SystemTime)
                        -> Result<Duration, Duration> {
            if self >= other {
                Ok(if self.t.tv_usec >= other.t.tv_usec {
                    Duration::new(self.t.tv_sec as u64 - other.t.tv_sec as u64,
                                  (self.t.tv_usec as u32 -
                                   other.t.tv_usec as u32) * 1000)
                } else {
                    Duration::new(self.t.tv_sec as u64 - 1 - other.t.tv_sec as u64,
                                  (self.t.tv_usec as u32 + (USEC_PER_SEC as u32) -
                                   other.t.tv_usec as u32) * 1000)
                })
            } else {
                match other.sub_time(self) {
                    Ok(d) => Err(d),
                    Err(d) => Ok(d),
                }
            }
        }

        pub fn add_duration(&self, other: &Duration) -> SystemTime {
            let secs = (self.t.tv_sec as i64).checked_add(other.as_secs() as i64);
            let mut secs = secs.expect("overflow when adding duration to time");

            // Nano calculations can't overflow because nanos are <1B which fit
            // in a u32.
            let mut usec = (other.subsec_nanos() / 1000) + self.t.tv_usec as u32;
            if usec > USEC_PER_SEC as u32 {
                usec -= USEC_PER_SEC as u32;
                secs = secs.checked_add(1).expect("overflow when adding \
                                                   duration to time");
            }
            SystemTime {
                t: libc::timeval {
                    tv_sec: secs as libc::time_t,
                    tv_usec: usec as libc::suseconds_t,
                },
            }
        }

        pub fn sub_duration(&self, other: &Duration) -> SystemTime {
            let secs = (self.t.tv_sec as i64).checked_sub(other.as_secs() as i64);
            let mut secs = secs.expect("overflow when subtracting duration \
                                        from time");

            // Similar to above, nanos can't overflow.
            let mut usec = self.t.tv_usec as i32 -
                           (other.subsec_nanos() / 1000) as i32;
            if usec < 0 {
                usec += USEC_PER_SEC as i32;
                secs = secs.checked_sub(1).expect("overflow when subtracting \
                                                   duration from time");
            }
            SystemTime {
                t: libc::timeval {
                    tv_sec: secs as libc::time_t,
                    tv_usec: usec as libc::suseconds_t,
                },
            }
        }
    }

    impl PartialEq for SystemTime {
        fn eq(&self, other: &SystemTime) -> bool {
            self.t.tv_sec == other.t.tv_sec && self.t.tv_usec == other.t.tv_usec
        }
    }

    impl Eq for SystemTime {}

    impl PartialOrd for SystemTime {
        fn partial_cmp(&self, other: &SystemTime) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for SystemTime {
        fn cmp(&self, other: &SystemTime) -> Ordering {
            let me = (self.t.tv_sec, self.t.tv_usec);
            let other = (other.t.tv_sec, other.t.tv_usec);
            me.cmp(&other)
        }
    }

    impl fmt::Debug for SystemTime {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("SystemTime")
             .field("tv_sec", &self.t.tv_sec)
             .field("tv_usec", &self.t.tv_usec)
             .finish()
        }
    }

    fn dur2intervals(dur: &Duration) -> u64 {
        let info = info();
        let nanos = dur.as_secs().checked_mul(NSEC_PER_SEC).and_then(|nanos| {
            nanos.checked_add(dur.subsec_nanos() as u64)
        }).expect("overflow converting duration to nanoseconds");
        mul_div_u64(nanos, info.denom as u64, info.numer as u64)
    }

    fn info() -> &'static libc::mach_timebase_info {
        static mut INFO: libc::mach_timebase_info = libc::mach_timebase_info {
            numer: 0,
            denom: 0,
        };
        static ONCE: Once = Once::new();

        unsafe {
            ONCE.call_once(|| {
                libc::mach_timebase_info(&mut INFO);
            });
            &INFO
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod inner {
    use cmp::Ordering;
    use fmt;
    use libc;
    use super::NSEC_PER_SEC;
    use sys::cvt;
    use time::Duration;

    #[derive(Copy, Clone)]
    struct Timespec {
        t: libc::timespec,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Instant {
        t: Timespec,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct SystemTime {
        t: Timespec,
    }

    pub const UNIX_EPOCH: SystemTime = SystemTime {
        t: Timespec {
            t: libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
        },
    };

    impl Instant {
        pub fn now() -> Instant {
            Instant { t: Timespec::now(libc::CLOCK_MONOTONIC) }
        }

        pub fn sub_instant(&self, other: &Instant) -> Duration {
            self.t.sub_timespec(&other.t).unwrap_or_else(|_| {
                panic!("other was less than the current instant")
            })
        }

        pub fn add_duration(&self, other: &Duration) -> Instant {
            Instant { t: self.t.add_duration(other) }
        }

        pub fn sub_duration(&self, other: &Duration) -> Instant {
            Instant { t: self.t.sub_duration(other) }
        }
    }

    impl fmt::Debug for Instant {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("Instant")
             .field("tv_sec", &self.t.t.tv_sec)
             .field("tv_nsec", &self.t.t.tv_nsec)
             .finish()
        }
    }

    impl SystemTime {
        pub fn now() -> SystemTime {
            SystemTime { t: Timespec::now(libc::CLOCK_REALTIME) }
        }

        pub fn sub_time(&self, other: &SystemTime)
                        -> Result<Duration, Duration> {
            self.t.sub_timespec(&other.t)
        }

        pub fn add_duration(&self, other: &Duration) -> SystemTime {
            SystemTime { t: self.t.add_duration(other) }
        }

        pub fn sub_duration(&self, other: &Duration) -> SystemTime {
            SystemTime { t: self.t.sub_duration(other) }
        }
    }

    impl fmt::Debug for SystemTime {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("SystemTime")
             .field("tv_sec", &self.t.t.tv_sec)
             .field("tv_nsec", &self.t.t.tv_nsec)
             .finish()
        }
    }

    impl Timespec {
        pub fn now(clock: libc::c_int) -> Timespec {
            let mut t = Timespec {
                t: libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }
            };
            cvt(unsafe {
                libc::clock_gettime(clock, &mut t.t)
            }).unwrap();
            t
        }

        fn sub_timespec(&self, other: &Timespec) -> Result<Duration, Duration> {
            if self >= other {
                Ok(if self.t.tv_nsec >= other.t.tv_nsec {
                    Duration::new((self.t.tv_sec - other.t.tv_sec) as u64,
                                  (self.t.tv_nsec - other.t.tv_nsec) as u32)
                } else {
                    Duration::new((self.t.tv_sec - 1 - other.t.tv_sec) as u64,
                                  self.t.tv_nsec as u32 + (NSEC_PER_SEC as u32) -
                                  other.t.tv_nsec as u32)
                })
            } else {
                match other.sub_timespec(self) {
                    Ok(d) => Err(d),
                    Err(d) => Ok(d),
                }
            }
        }

        fn add_duration(&self, other: &Duration) -> Timespec {
            let secs = (self.t.tv_sec as i64).checked_add(other.as_secs() as i64);
            let mut secs = secs.expect("overflow when adding duration to time");

            // Nano calculations can't overflow because nanos are <1B which fit
            // in a u32.
            let mut nsec = other.subsec_nanos() + self.t.tv_nsec as u32;
            if nsec > NSEC_PER_SEC as u32 {
                nsec -= NSEC_PER_SEC as u32;
                secs = secs.checked_add(1).expect("overflow when adding \
                                                   duration to time");
            }
            Timespec {
                t: libc::timespec {
                    tv_sec: secs as libc::time_t,
                    tv_nsec: nsec as libc::c_long,
                },
            }
        }

        fn sub_duration(&self, other: &Duration) -> Timespec {
            let secs = (self.t.tv_sec as i64).checked_sub(other.as_secs() as i64);
            let mut secs = secs.expect("overflow when subtracting duration \
                                        from time");

            // Similar to above, nanos can't overflow.
            let mut nsec = self.t.tv_nsec as i32 - other.subsec_nanos() as i32;
            if nsec < 0 {
                nsec += NSEC_PER_SEC as i32;
                secs = secs.checked_sub(1).expect("overflow when subtracting \
                                                   duration from time");
            }
            Timespec {
                t: libc::timespec {
                    tv_sec: secs as libc::time_t,
                    tv_nsec: nsec as libc::c_long,
                },
            }
        }
    }

    impl PartialEq for Timespec {
        fn eq(&self, other: &Timespec) -> bool {
            self.t.tv_sec == other.t.tv_sec && self.t.tv_nsec == other.t.tv_nsec
        }
    }

    impl Eq for Timespec {}

    impl PartialOrd for Timespec {
        fn partial_cmp(&self, other: &Timespec) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Timespec {
        fn cmp(&self, other: &Timespec) -> Ordering {
            let me = (self.t.tv_sec, self.t.tv_nsec);
            let other = (other.t.tv_sec, other.t.tv_nsec);
            me.cmp(&other)
        }
    }
}
