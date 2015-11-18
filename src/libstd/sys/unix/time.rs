// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::inner::SteadyTime;

const NSEC_PER_SEC: u64 = 1_000_000_000;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod inner {
    use libc;
    use time::Duration;
    use ops::Sub;
    use sync::Once;
    use super::NSEC_PER_SEC;

    pub struct SteadyTime {
        t: u64
    }

    impl SteadyTime {
        pub fn now() -> SteadyTime {
            SteadyTime {
                t: unsafe { libc::mach_absolute_time() },
            }
        }
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

    #[unstable(feature = "libstd_sys_internals", issue = "0")]
    impl<'a> Sub for &'a SteadyTime {
        type Output = Duration;

        fn sub(self, other: &SteadyTime) -> Duration {
            let info = info();
            let diff = self.t as u64 - other.t as u64;
            let nanos = diff * info.numer as u64 / info.denom as u64;
            Duration::new(nanos / NSEC_PER_SEC, (nanos % NSEC_PER_SEC) as u32)
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod inner {
    use libc;
    use time::Duration;
    use ops::Sub;
    use super::NSEC_PER_SEC;

    pub struct SteadyTime {
        t: libc::timespec,
    }

    // Apparently android provides this in some other library?
    // Bitrig's RT extensions are in the C library, not a separate librt
    // OpenBSD and NaCl provide it via libc
    #[cfg(not(any(target_os = "android",
                  target_os = "bitrig",
                  target_os = "netbsd",
                  target_os = "openbsd",
                  target_env = "musl",
                  target_os = "nacl")))]
    #[link(name = "rt")]
    extern {}

    impl SteadyTime {
        pub fn now() -> SteadyTime {
            let mut t = SteadyTime {
                t: libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }
            };
            unsafe {
                assert_eq!(0, libc::clock_gettime(libc::CLOCK_MONOTONIC,
                                                  &mut t.t));
            }
            t
        }
    }

    #[unstable(feature = "libstd_sys_internals", issue = "0")]
    impl<'a> Sub for &'a SteadyTime {
        type Output = Duration;

        fn sub(self, other: &SteadyTime) -> Duration {
            if self.t.tv_nsec >= other.t.tv_nsec {
                Duration::new(self.t.tv_sec as u64 - other.t.tv_sec as u64,
                              self.t.tv_nsec as u32 - other.t.tv_nsec as u32)
            } else {
                Duration::new(self.t.tv_sec as u64 - 1 - other.t.tv_sec as u64,
                              self.t.tv_nsec as u32 + (NSEC_PER_SEC as u32) -
                                          other.t.tv_nsec as u32)
            }
        }
    }
}
