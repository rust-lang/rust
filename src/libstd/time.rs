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

use crate::cmp;
use crate::error::Error;
use crate::fmt;
use crate::ops::{Add, Sub, AddAssign, SubAssign};
use crate::sys::time;
use crate::sys_common::FromInner;
use crate::sys_common::mutex::Mutex;

#[stable(feature = "time", since = "1.3.0")]
pub use core::time::Duration;

/// A measurement of a monotonically nondecreasing clock.
/// Opaque and useful only with `Duration`.
///
/// Instants are always guaranteed to be no less than any previously measured
/// instant when created, and are often useful for tasks such as measuring
/// benchmarks or timing how long an operation takes.
///
/// Note, however, that instants are not guaranteed to be **steady**. In other
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
/// The size of an `Instant` struct may vary depending on the target operating
/// system.
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
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
/// The size of a `SystemTime` struct may vary depending on the target operating
/// system.
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
///            // an error occurred!
///            println!("Error: {:?}", e);
///        }
///    }
/// }
/// ```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        let os_now = time::Instant::now();

        // And here we come upon a sad state of affairs. The whole point of
        // `Instant` is that it's monotonically increasing. We've found in the
        // wild, however, that it's not actually monotonically increasing for
        // one reason or another. These appear to be OS and hardware level bugs,
        // and there's not really a whole lot we can do about them. Here's a
        // taste of what we've found:
        //
        // * #48514 - OpenBSD, x86_64
        // * #49281 - linux arm64 and s390x
        // * #51648 - windows, x86
        // * #56560 - windows, x86_64, AWS
        // * #56612 - windows, x86, vm (?)
        // * #56940 - linux, arm64
        // * https://bugzilla.mozilla.org/show_bug.cgi?id=1487778 - a similar
        //   Firefox bug
        //
        // It simply seems that this it just happens so that a lot in the wild
        // we're seeing panics across various platforms where consecutive calls
        // to `Instant::now`, such as via the `elapsed` function, are panicking
        // as they're going backwards. Placed here is a last-ditch effort to try
        // to fix things up. We keep a global "latest now" instance which is
        // returned instead of what the OS says if the OS goes backwards.
        //
        // To hopefully mitigate the impact of this though a few platforms are
        // whitelisted as "these at least haven't gone backwards yet".
        if time::Instant::actually_monotonic() {
            return Instant(os_now)
        }

        static LOCK: Mutex = Mutex::new();
        static mut LAST_NOW: time::Instant = time::Instant::zero();
        unsafe {
            let _lock = LOCK.lock();
            let now = cmp::max(LAST_NOW, os_now);
            LAST_NOW = now;
            Instant(now)
        }
    }

    /// Returns the amount of time elapsed from another instant to this one.
    ///
    /// # Panics
    ///
    /// This function will panic if `earlier` is later than `self`.
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
        self.0.checked_sub_instant(&earlier.0).expect("supplied instant is later than self")
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or None if that instant is earlier than this one.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(checked_duration_since)]
    /// use std::time::{Duration, Instant};
    /// use std::thread::sleep;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.checked_duration_since(now));
    /// println!("{:?}", now.checked_duration_since(new_now)); // None
    /// ```
    #[unstable(feature = "checked_duration_since", issue = "58402")]
    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.0.checked_sub_instant(&earlier.0)
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is earlier than this one.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(checked_duration_since)]
    /// use std::time::{Duration, Instant};
    /// use std::thread::sleep;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.saturating_duration_since(now));
    /// println!("{:?}", now.saturating_duration_since(new_now)); // 0ns
    /// ```
    #[unstable(feature = "checked_duration_since", issue = "58402")]
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or(Duration::new(0, 0))
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

    /// Returns `Some(t)` where `t` is the time `self + duration` if `t` can be represented as
    /// `Instant` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    #[stable(feature = "time_checked_add", since = "1.34.0")]
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_add_duration(&duration).map(Instant)
    }

    /// Returns `Some(t)` where `t` is the time `self - duration` if `t` can be represented as
    /// `Instant` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    #[stable(feature = "time_checked_add", since = "1.34.0")]
    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_sub_duration(&duration).map(Instant)
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Add<Duration> for Instant {
    type Output = Instant;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`checked_add`] for a version without panic.
    ///
    /// [`checked_add`]: ../../std/time/struct.Instant.html#method.checked_add
    fn add(self, other: Duration) -> Instant {
        self.checked_add(other)
            .expect("overflow when adding duration to instant")
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
        self.checked_sub(other)
            .expect("overflow when subtracting duration from instant")
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl SystemTime {
    /// An anchor in time which can be used to create new `SystemTime` instances or
    /// learn about where in time a `SystemTime` lies.
    ///
    /// This constant is defined to be "1970-01-01 00:00:00 UTC" on all systems with
    /// respect to the system clock. Using `duration_since` on an existing
    /// `SystemTime` instance can tell how far away from this point in time a
    /// measurement lies, and using `UNIX_EPOCH + duration` can be used to create a
    /// `SystemTime` instance to represent another fixed point in time.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::SystemTime;
    ///
    /// match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
    ///     Ok(n) => println!("1970-01-01 00:00:00 UTC was {} seconds ago!", n.as_secs()),
    ///     Err(_) => panic!("SystemTime before UNIX EPOCH!"),
    /// }
    /// ```
    #[stable(feature = "assoc_unix_epoch", since = "1.28.0")]
    pub const UNIX_EPOCH: SystemTime = UNIX_EPOCH;

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
    /// drift and updates (e.g., the system clock could go backwards), so this
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

    /// Returns `Some(t)` where `t` is the time `self + duration` if `t` can be represented as
    /// `SystemTime` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    #[stable(feature = "time_checked_add", since = "1.34.0")]
    pub fn checked_add(&self, duration: Duration) -> Option<SystemTime> {
        self.0.checked_add_duration(&duration).map(SystemTime)
    }

    /// Returns `Some(t)` where `t` is the time `self - duration` if `t` can be represented as
    /// `SystemTime` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    #[stable(feature = "time_checked_add", since = "1.34.0")]
    pub fn checked_sub(&self, duration: Duration) -> Option<SystemTime> {
        self.0.checked_sub_duration(&duration).map(SystemTime)
    }
}

#[stable(feature = "time2", since = "1.8.0")]
impl Add<Duration> for SystemTime {
    type Output = SystemTime;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`checked_add`] for a version without panic.
    ///
    /// [`checked_add`]: ../../std/time/struct.SystemTime.html#method.checked_add
    fn add(self, dur: Duration) -> SystemTime {
        self.checked_add(dur)
            .expect("overflow when adding duration to instant")
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
        self.checked_sub(dur)
            .expect("overflow when subtracting duration from instant")
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
///
/// # Examples
///
/// ```no_run
/// use std::time::{SystemTime, UNIX_EPOCH};
///
/// match SystemTime::now().duration_since(UNIX_EPOCH) {
///     Ok(n) => println!("1970-01-01 00:00:00 UTC was {} seconds ago!", n.as_secs()),
///     Err(_) => panic!("SystemTime before UNIX EPOCH!"),
/// }
/// ```
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                assert!(a - Duration::new(0, 1000) <= b,
                        "{:?} is not almost equal to {:?}", a, b);
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
        println!("a: {:?}", a);
        println!("b: {:?}", b);
        let dur = b.duration_since(a);
        println!("dur: {:?}", dur);
        assert_almost_eq!(b - dur, a);
        assert_almost_eq!(a + dur, b);

        let second = Duration::new(1, 0);
        assert_almost_eq!(a - second + second, a);
        assert_almost_eq!(a.checked_sub(second).unwrap().checked_add(second).unwrap(), a);

        // checked_add_duration will not panic on overflow
        let mut maybe_t = Some(Instant::now());
        let max_duration = Duration::from_secs(u64::max_value());
        // in case `Instant` can store `>= now + max_duration`.
        for _ in 0..2 {
            maybe_t = maybe_t.and_then(|t| t.checked_add(max_duration));
        }
        assert_eq!(maybe_t, None);

        // checked_add_duration calculates the right time and will work for another year
        let year = Duration::from_secs(60 * 60 * 24 * 365);
        assert_eq!(a + year, a.checked_add(year).unwrap());
    }

    #[test]
    fn instant_math_is_associative() {
        let now = Instant::now();
        let offset = Duration::from_millis(5);
        // Changing the order of instant math shouldn't change the results,
        // especially when the expression reduces to X + identity.
        assert_eq!((now + offset) - now, (now - now) + offset);
    }

    #[test]
    #[should_panic]
    fn instant_duration_since_panic() {
        let a = Instant::now();
        (a - Duration::new(1, 0)).duration_since(a);
    }

    #[test]
    fn instant_checked_duration_since_nopanic() {
        let now = Instant::now();
        let earlier = now - Duration::new(1, 0);
        let later = now + Duration::new(1, 0);
        assert_eq!(earlier.checked_duration_since(now), None);
        assert_eq!(later.checked_duration_since(now), Some(Duration::new(1, 0)));
        assert_eq!(now.checked_duration_since(now), Some(Duration::new(0, 0)));
    }

    #[test]
    fn instant_saturating_duration_since_nopanic() {
        let a = Instant::now();
        let ret = (a - Duration::new(1, 0)).saturating_duration_since(a);
        assert_eq!(ret, Duration::new(0,0));
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
                assert_almost_eq!(a - dur, b);
            }
        }

        let second = Duration::new(1, 0);
        assert_almost_eq!(a.duration_since(a - second).unwrap(), second);
        assert_almost_eq!(a.duration_since(a + second).unwrap_err()
                           .duration(), second);

        assert_almost_eq!(a - second + second, a);
        assert_almost_eq!(a.checked_sub(second).unwrap().checked_add(second).unwrap(), a);

        let one_second_from_epoch = UNIX_EPOCH + Duration::new(1, 0);
        let one_second_from_epoch2 = UNIX_EPOCH + Duration::new(0, 500_000_000)
            + Duration::new(0, 500_000_000);
        assert_eq!(one_second_from_epoch, one_second_from_epoch2);

        // checked_add_duration will not panic on overflow
        let mut maybe_t = Some(SystemTime::UNIX_EPOCH);
        let max_duration = Duration::from_secs(u64::max_value());
        // in case `SystemTime` can store `>= UNIX_EPOCH + max_duration`.
        for _ in 0..2 {
            maybe_t = maybe_t.and_then(|t| t.checked_add(max_duration));
        }
        assert_eq!(maybe_t, None);

        // checked_add_duration calculates the right time and will work for another year
        let year = Duration::from_secs(60 * 60 * 24 * 365);
        assert_eq!(a + year, a.checked_add(year).unwrap());
    }

    #[test]
    fn system_time_elapsed() {
        let a = SystemTime::now();
        drop(a.elapsed());
    }

    #[test]
    fn since_epoch() {
        let ts = SystemTime::now();
        let a = ts.duration_since(UNIX_EPOCH + Duration::new(1, 0)).unwrap();
        let b = ts.duration_since(UNIX_EPOCH).unwrap();
        assert!(b > a);
        assert_eq!(b - a, Duration::new(1, 0));

        let thirty_years = Duration::new(1, 0) * 60 * 60 * 24 * 365 * 30;

        // Right now for CI this test is run in an emulator, and apparently the
        // aarch64 emulator's sense of time is that we're still living in the
        // 70s.
        //
        // Otherwise let's assume that we're all running computers later than
        // 2000.
        if !cfg!(target_arch = "aarch64") {
            assert!(a > thirty_years);
        }

        // let's assume that we're all running computers earlier than 2090.
        // Should give us ~70 years to fix this!
        let hundred_twenty_years = thirty_years * 4;
        assert!(a < hundred_twenty_years);
    }
}
