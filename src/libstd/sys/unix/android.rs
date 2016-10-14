// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Android ABI-compatibility module
//!
//! The ABI of Android has changed quite a bit over time, and libstd attempts to
//! be both forwards and backwards compatible as much as possible. We want to
//! always work with the most recent version of Android, but we also want to
//! work with older versions of Android for whenever projects need to.
//!
//! Our current minimum supported Android version is `android-9`, e.g. Android
//! with API level 9. We then in theory want to work on that and all future
//! versions of Android!
//!
//! Some of the detection here is done at runtime via `dlopen` and
//! introspection. Other times no detection is performed at all and we just
//! provide a fallback implementation as some versions of Android we support
//! don't have the function.
//!
//! You'll find more details below about why each compatibility shim is needed.

#![cfg(target_os = "android")]

use libc::{c_int, c_void, sighandler_t, size_t, ssize_t};
use libc::{ftruncate, pread, pwrite};

use io;
use super::{cvt, cvt_r};

// The `log2` and `log2f` functions apparently appeared in android-18, or at
// least you can see they're not present in the android-17 header [1] and they
// are present in android-18 [2].
//
// [1]: https://chromium.googlesource.com/android_tools/+/20ee6d20/ndk/platforms
//                                       /android-17/arch-arm/usr/include/math.h
// [2]: https://chromium.googlesource.com/android_tools/+/20ee6d20/ndk/platforms
//                                       /android-18/arch-arm/usr/include/math.h
//
// Note that these shims are likely less precise than directly calling `log2`,
// but hopefully that should be enough for now...
//
// Note that mathematically, for any arbitrary `y`:
//
//      log_2(x) = log_y(x) / log_y(2)
//               = log_y(x) / (1 / log_2(y))
//               = log_y(x) * log_2(y)
//
// Hence because `ln` (log_e) is available on all Android we just choose `y = e`
// and get:
//
//      log_2(x) = ln(x) * log_2(e)

#[cfg(not(test))]
pub fn log2f32(f: f32) -> f32 {
    f.ln() * ::f32::consts::LOG2_E
}

#[cfg(not(test))]
pub fn log2f64(f: f64) -> f64 {
    f.ln() * ::f64::consts::LOG2_E
}

// Back in the day [1] the `signal` function was just an inline wrapper
// around `bsd_signal`, but starting in API level android-20 the `signal`
// symbols was introduced [2]. Finally, in android-21 the API `bsd_signal` was
// removed [3].
//
// Basically this means that if we want to be binary compatible with multiple
// Android releases (oldest being 9 and newest being 21) then we need to check
// for both symbols and not actually link against either.
//
// [1]: https://chromium.googlesource.com/android_tools/+/20ee6d20/ndk/platforms
//                                       /android-18/arch-arm/usr/include/signal.h
// [2]: https://chromium.googlesource.com/android_tools/+/fbd420/ndk_experimental
//                                       /platforms/android-20/arch-arm
//                                       /usr/include/signal.h
// [3]: https://chromium.googlesource.com/android_tools/+/20ee6d/ndk/platforms
//                                       /android-21/arch-arm/usr/include/signal.h
pub unsafe fn signal(signum: c_int, handler: sighandler_t) -> sighandler_t {
    weak!(fn signal(c_int, sighandler_t) -> sighandler_t);
    weak!(fn bsd_signal(c_int, sighandler_t) -> sighandler_t);

    let f = signal.get().or_else(|| bsd_signal.get());
    let f = f.expect("neither `signal` nor `bsd_signal` symbols found");
    f(signum, handler)
}

// The `ftruncate64` symbol apparently appeared in android-12, so we do some
// dynamic detection to see if we can figure out whether `ftruncate64` exists.
//
// If it doesn't we just fall back to `ftruncate`, generating an error for
// too-large values.
#[cfg(target_pointer_width = "32")]
pub fn ftruncate64(fd: c_int, size: u64) -> io::Result<()> {
    weak!(fn ftruncate64(c_int, i64) -> c_int);

    unsafe {
        match ftruncate64.get() {
            Some(f) => cvt_r(|| f(fd, size as i64)).map(|_| ()),
            None => {
                if size > i32::max_value() as u64 {
                    Err(io::Error::new(io::ErrorKind::InvalidInput,
                                       "cannot truncate >2GB"))
                } else {
                    cvt_r(|| ftruncate(fd, size as i32)).map(|_| ())
                }
            }
        }
    }
}

#[cfg(target_pointer_width = "64")]
pub fn ftruncate64(fd: c_int, size: u64) -> io::Result<()> {
    unsafe {
        cvt_r(|| ftruncate(fd, size as i64)).map(|_| ())
    }
}

#[cfg(target_pointer_width = "32")]
pub unsafe fn cvt_pread64(fd: c_int, buf: *mut c_void, count: size_t, offset: i64)
    -> io::Result<ssize_t>
{
    use convert::TryInto;
    weak!(fn pread64(c_int, *mut c_void, size_t, i64) -> ssize_t);
    pread64.get().map(|f| cvt(f(fd, buf, count, offset))).unwrap_or_else(|| {
        if let Ok(o) = offset.try_into() {
            cvt(pread(fd, buf, count, o))
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidInput,
                               "cannot pread >2GB"))
        }
    })
}

#[cfg(target_pointer_width = "32")]
pub unsafe fn cvt_pwrite64(fd: c_int, buf: *const c_void, count: size_t, offset: i64)
    -> io::Result<ssize_t>
{
    use convert::TryInto;
    weak!(fn pwrite64(c_int, *const c_void, size_t, i64) -> ssize_t);
    pwrite64.get().map(|f| cvt(f(fd, buf, count, offset))).unwrap_or_else(|| {
        if let Ok(o) = offset.try_into() {
            cvt(pwrite(fd, buf, count, o))
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidInput,
                               "cannot pwrite >2GB"))
        }
    })
}

#[cfg(target_pointer_width = "64")]
pub unsafe fn cvt_pread64(fd: c_int, buf: *mut c_void, count: size_t, offset: i64)
    -> io::Result<ssize_t>
{
    cvt(pread(fd, buf, count, offset))
}

#[cfg(target_pointer_width = "64")]
pub unsafe fn cvt_pwrite64(fd: c_int, buf: *const c_void, count: size_t, offset: i64)
    -> io::Result<ssize_t>
{
    cvt(pwrite(fd, buf, count, offset))
}
