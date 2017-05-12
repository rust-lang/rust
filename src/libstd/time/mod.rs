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
//!
//! Example:
//!
//! ```
//! use std::time::Duration;
//!
//! let five_seconds = Duration::new(5, 0);
//! // both declarations are equivalent
//! assert_eq!(Duration::new(5, 0), Duration::from_secs(5));
//! ```

#![stable(feature = "time", since = "1.3.0")]

use error::Error;
use fmt;
use ops::{Add, Sub, AddAssign, SubAssign};
use sys::time;
use sys_common::FromInner;

#[stable(feature = "time", since = "1.3.0")]
pub use self::duration::Duration;

mod duration;

/// A measurement of a monotonically increasing clock.
/// Opaque and useful only with `Duration`.
///
/// Instants are always guaranteed to be greater than any previously measured
/// instant when created, and are often useful for tasks such as measuring
/// benchmarks or timing how long an operation takes.
///
/// Note, however, that instants are not guaranteed to be **steady**.  In other
/// words, each tick of the underlying clock may not be the same length (e.g.
/// some seconds may be longer than others). An instant may jump forwards or
/// experience time dilation (slow down or speed up), but it will never go
/// backwards.
///
/// Instants are opaque types that can only be compared to one another. There is
/// no method to get "the number of seconds" from an instant. Instead, it only
/// allows measuring the duration between two instants (or comparing two
/// instants).
///
/// Example:
///
/// ```no_run
/// use std::time::{Duration, Instant};
/// use std::thread::sleep;
///
/// fn main() {
///    let now = Instant::now();
///
///    // we sleep for 2 seconds
///    sleep(Duration::new(2, 0));
///    // it prints '2'
///    println!("{}", now.elapsed().as_secs());
/// }
/// ```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[stable(feature = "time2", since = "1.8.0")]
pub struct Instant(time::Instant);

/// A measurement of the system clock, useful for talking to
/// external entities like the file system or other processes.
///
/// Distinct from the [`Instant`] type, this time measurement **is not
/// monotonic**. This means that you can save a file to the file system, then
/// save another file to the file system, **and the second file has a
/// `SystemTime` measurement earlier than the first**. In other words, an
/// operation that happens after another operation in real time may have an
/// earlier `SystemTime`!
///
/// Consequently, comparing two `SystemTime` instances to learn about the
/// duration between them returns a [`Result`] instead of an infallible [`Duration`]
/// to indicate that this sort of time drift may happen and needs to be handled.
///
/// Although a `SystemTime` cannot be directly inspected, the [`UNIX_EPOCH`]
/// constant is provided in this module as an anchor in time to learn
/// information about a `SystemTime`. By calculating the duration from this
/// fixed point in time, a `SystemTime` can be converted to a human-readable time,
/// or perhaps some other string representation.
///
/// [`Instant`]: ../../std/time/struct.Instant.html
/// [`Result`]: ../../std/result/enum.Result.html
/// [`Duration`]: ../../std/time/struct.Duration.html
/// [`UNIX_EPOCH`]: ../../std/time/constant.UNIX_EPOCH.html
///
/// Example:
///
/// ```no_run
/// use std::time::{Duration, SystemTime};
/// use std::thread::sleep;
///
/// fn main() {
///    let now = SystemTime::now();
///
///    // we sleep for 2 seconds
///    sleep(Duration::new(2, 0));
///    match now.elapsed() {
///        Ok(elapsed) => {
///            // it prints '2'
///            println!("{}", elapsed.as_secs());
///        }
///        Err(e) => {
///            // an error occured!
///            println!("Error: {:?}", e);
///        }
///    }
/// }
/// ```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[stable(feature = "time2", since = "1.8.0")]
pub struct SystemTime(time::SystemTime);

/// An error returned from the `duration_since` and `elapsed` methods on
/// `SystemTime`, used to learn how far in the opposite direction a system time
/// lies.
///
/// # Examples
///
/// ```no_run
/// use std::thread::sleep;
/// use std::time::{Duration, SystemTime};
///
/// let sys_time = SystemTime::now();
/// sleep(Duration::from_secs(1));
/// let new_sys_time = SystemTime::now();
/// match sys_time.duration_since(new_sys_time) {
///     Ok(_) => {}
///     Err(e) => println!("SystemTimeError difference: {:?}", e.duration()),
/// }
/// ```
#[derive(Clone, Debug)]
#[stable(feature = "time2", since = "1.8.0")]
pub struct SystemTimeError(Duration);

impl Instant {
    /// Returns an instant corresponding to "now".
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// let now = Instant::now();
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn now() -> Instant {
        Instant(time::Instant::now())
    }

    /// Returns the amount of time elapsed from another instant to this one.
    ///
    /// # Panics
    ///
    /// This function will panic if `earlier` is later than `self`, which should
    /// only be possible if `earlier` was created after `self`. Because
    /// `Instant` is monotonic, the only time that this should happen should be
    /// a bug.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::{Duration, Instant};
    /// use std::thread::sleep;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.duration_since(now));
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.0.sub_instant(&earlier.0)
    }

    /// Returns the amount of time elapsed since this instant was created.
    ///
    /// # Panics
    ///
    /// This function may panic if the current time is earlier than this
    /// instant, which is something that can happen if an `Instant` is
    /// produced synthetically.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::thread::sleep;
    /// use std::time::{Duration, Instant};
    ///
    /// let instant = Instant::now();
    /// let three_secs = Duration::from_secs(3);
    /// sleep(three_secs);
    /// assert!(instant.elapsed() >= three_secs);
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn elapsed(&self) -> Duration {
        Instant::now() - *self
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        Instant(self.0.add_duration(&other))
    }
}

#[stable(feature = "time_augmented_assignment", since = "1.9.0")]
impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        Instant(self.0.sub_duration(&other))
    }
}

#[stable(feature = "time_augmented_assignment", since = "1.9.0")]
impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Sub<Instant> for Instant {
    type Output = Duration;

    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl fmt::Debug for Instant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl SystemTime {
    /// Returns the system time corresponding to "now".
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::SystemTime;
    ///
    /// let sys_time = SystemTime::now();
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn now() -> SystemTime {
        SystemTime(time::SystemTime::now())
    }

    /// Returns the amount of time elapsed from an earlier point in time.
    ///
    /// This function may fail because measurements taken earlier are not
    /// guaranteed to always be before later measurements (due to anomalies such
    /// as the system clock being adjusted either forwards or backwards).
    ///
    /// If successful, [`Ok`]`(`[`Duration`]`)` is returned where the duration represents
    /// the amount of time elapsed from the specified measurement to this one.
    ///
    /// Returns an [`Err`] if `earlier` is later than `self`, and the error
    /// contains how far from `self` the time is.
    ///
    /// [`Ok`]: ../../std/result/enum.Result.html#variant.Ok
    /// [`Duration`]: ../../std/time/struct.Duration.html
    /// [`Err`]: ../../std/result/enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::SystemTime;
    ///
    /// let sys_time = SystemTime::now();
    /// let difference = sys_time.duration_since(sys_time)
    ///                          .expect("SystemTime::duration_since failed");
    /// println!("{:?}", difference);
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn duration_since(&self, earlier: SystemTime)
                          -> Result<Duration, SystemTimeError> {
        self.0.sub_time(&earlier.0).map_err(SystemTimeError)
    }

    /// Returns the amount of time elapsed since this system time was created.
    ///
    /// This function may fail as the underlying system clock is susceptible to
    /// drift and updates (e.g. the system clock could go backwards), so this
    /// function may not always succeed. If successful, [`Ok`]`(`[`Duration`]`)` is
    /// returned where the duration represents the amount of time elapsed from
    /// this time measurement to the current time.
    ///
    /// Returns an [`Err`] if `self` is later than the current system time, and
    /// the error contains how far from the current system time `self` is.
    ///
    /// [`Ok`]: ../../std/result/enum.Result.html#variant.Ok
    /// [`Duration`]: ../../std/time/struct.Duration.html
    /// [`Err`]: ../../std/result/enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::thread::sleep;
    /// use std::time::{Duration, SystemTime};
    ///
    /// let sys_time = SystemTime::now();
    /// let one_sec = Duration::from_secs(1);
    /// sleep(one_sec);
    /// assert!(sys_time.elapsed().unwrap() >= one_sec);
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn elapsed(&self) -> Result<Duration, SystemTimeError> {
        SystemTime::now().duration_since(*self)
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Add<Duration> for SystemTime {
    type Output = SystemTime;

    fn add(self, dur: Duration) -> SystemTime {
        SystemTime(self.0.add_duration(&dur))
    }
}

#[stable(feature = "time_augmented_assignment", since = "1.9.0")]
impl AddAssign<Duration> for SystemTime {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Sub<Duration> for SystemTime {
    type Output = SystemTime;

    fn sub(self, dur: Duration) -> SystemTime {
        SystemTime(self.0.sub_duration(&dur))
    }
}

#[stable(feature = "time_augmented_assignment", since = "1.9.0")]
impl SubAssign<Duration> for SystemTime {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl fmt::Debug for SystemTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// An anchor in time which can be used to create new `SystemTime` instances or
/// learn about where in time a `SystemTime` lies.
///
/// This constant is defined to be "1970-01-01 00:00:00 UTC" on all systems with
/// respect to the system clock. Using `duration_since` on an existing
/// [`SystemTime`] instance can tell how far away from this point in time a
/// measurement lies, and using `UNIX_EPOCH + duration` can be used to create a
/// [`SystemTime`] instance to represent another fixed point in time.
///
/// [`SystemTime`]: ../../std/time/struct.SystemTime.html
#[stable(feature = "time2", since = "1.8.0")]
pub const UNIX_EPOCH: SystemTime = SystemTime(time::UNIX_EPOCH);

impl SystemTimeError {
    /// Returns the positive duration which represents how far forward the
    /// second system time was from the first.
    ///
    /// A `SystemTimeError` is returned from the [`duration_since`] and [`elapsed`]
    /// methods of [`SystemTime`] whenever the second system time represents a point later
    /// in time than the `self` of the method call.
    ///
    /// [`duration_since`]: ../../std/time/struct.SystemTime.html#method.duration_since
    /// [`elapsed`]: ../../std/time/struct.SystemTime.html#method.elapsed
    /// [`SystemTime`]: ../../std/time/struct.SystemTime.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::thread::sleep;
    /// use std::time::{Duration, SystemTime};
    ///
    /// let sys_time = SystemTime::now();
    /// sleep(Duration::from_secs(1));
    /// let new_sys_time = SystemTime::now();
    /// match sys_time.duration_since(new_sys_time) {
    ///     Ok(_) => {}
    ///     Err(e) => println!("SystemTimeError difference: {:?}", e.duration()),
    /// }
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn duration(&self) -> Duration {
        self.0
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Error for SystemTimeError {
    fn description(&self) -> &str { "other time was not earlier than self" }
}

#[stable(feature = "time2", since = "1.8.0")]
impl fmt::Display for SystemTimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "second time provided was later than self")
    }
}

impl FromInner<time::SystemTime> for SystemTime {
    fn from_inner(time: time::SystemTime) -> SystemTime {
        SystemTime(time)
    }
}

#[cfg(test)]
mod tests {
    use super::{Instant, SystemTime, Duration, UNIX_EPOCH};

    macro_rules! assert_almost_eq {
        ($a:expr, $b:expr) => ({
            let (a, b) = ($a, $b);
            if a != b {
                let (a, b) = if a > b {(a, b)} else {(b, a)};
                assert!(a - Duration::new(0, 100) <= b);
            }
        })
    }

    #[test]
    fn instant_monotonic() {
        let a = Instant::now();
        let b = Instant::now();
        assert!(b >= a);
    }

    #[test]
    fn instant_elapsed() {
        let a = Instant::now();
        a.elapsed();
    }

    #[test]
    fn instant_math() {
        let a = Instant::now();
        let b = Instant::now();
        let dur = b.duration_since(a);
        assert_almost_eq!(b - dur, a);
        assert_almost_eq!(a + dur, b);

        let second = Duration::new(1, 0);
        assert_almost_eq!(a - second + second, a);
    }

    #[test]
    #[should_panic]
    fn instant_duration_panic() {
        let a = Instant::now();
        (a - Duration::new(1, 0)).duration_since(a);
    }

    #[test]
    fn system_time_math() {
        let a = SystemTime::now();
        let b = SystemTime::now();
        match b.duration_since(a) {
            Ok(dur) if dur == Duration::new(0, 0) => {
                assert_almost_eq!(a, b);
            }
            Ok(dur) => {
                assert!(b > a);
                assert_almost_eq!(b - dur, a);
                assert_almost_eq!(a + dur, b);
            }
            Err(dur) => {
                let dur = dur.duration();
                assert!(a > b);
                assert_almost_eq!(b + dur, a);
                assert_almost_eq!(b - dur, a);
            }
        }

        let second = Duration::new(1, 0);
        assert_almost_eq!(a.duration_since(a - second).unwrap(), second);
        assert_almost_eq!(a.duration_since(a + second).unwrap_err()
                           .duration(), second);

        assert_almost_eq!(a - second + second, a);

        let eighty_years = second * 60 * 60 * 24 * 365 * 80;
        assert_almost_eq!(a - eighty_years + eighty_years, a);
        assert_almost_eq!(a - (eighty_years * 10) + (eighty_years * 10), a);

        let one_second_from_epoch = UNIX_EPOCH + Duration::new(1, 0);
        let one_second_from_epoch2 = UNIX_EPOCH + Duration::new(0, 500_000_000)
            + Duration::new(0, 500_000_000);
        assert_eq!(one_second_from_epoch, one_second_from_epoch2);
    }

    #[test]
    fn system_time_elapsed() {
        let a = SystemTime::now();
        drop(a.elapsed());
    }

    #[test]
    fn since_epoch() {
        let ts = SystemTime::now();
        let a = ts.duration_since(UNIX_EPOCH).unwrap();
        let b = ts.duration_since(UNIX_EPOCH - Duration::new(1, 0)).unwrap();
        assert!(b > a);
        assert_eq!(b - a, Duration::new(1, 0));

        // let's assume that we're all running computers later than 2000
        let thirty_years = Duration::new(1, 0) * 60 * 60 * 24 * 365 * 30;
        assert!(a > thirty_years);

        // let's assume that we're all running computers earlier than 2090.
        // Should give us ~70 years to fix this!
        let hundred_twenty_years = thirty_years * 4;
        assert!(a < hundred_twenty_years);
    }
}
