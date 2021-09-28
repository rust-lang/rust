//! Temporal quantification.
//!
//! # Examples:
//!
//! There are multiple ways to create a new [`Duration`]:
//!
//! ```
//! # use std::time::Duration;
//! let five_seconds = Duration::from_secs(5);
//! assert_eq!(five_seconds, Duration::from_millis(5_000));
//! assert_eq!(five_seconds, Duration::from_micros(5_000_000));
//! assert_eq!(five_seconds, Duration::from_nanos(5_000_000_000));
//!
//! let ten_seconds = Duration::from_secs(10);
//! let seven_nanos = Duration::from_nanos(7);
//! let total = ten_seconds + seven_nanos;
//! assert_eq!(total, Duration::new(10, 7));
//! ```
//!
//! Using [`Instant`] to calculate how long a function took to run:
//!
//! ```ignore (incomplete)
//! let now = Instant::now();
//!
//! // Calling a slow function, it may take a while
//! slow_function();
//!
//! let elapsed_time = now.elapsed();
//! println!("Running slow_function() took {} seconds.", elapsed_time.as_secs());
//! ```

#![stable(feature = "time", since = "1.3.0")]

mod monotonic;
#[cfg(test)]
mod tests;

use crate::error::Error;
use crate::fmt;
use crate::ops::{Add, AddAssign, Sub, SubAssign};
use crate::sys::time;
use crate::sys_common::FromInner;

#[stable(feature = "time", since = "1.3.0")]
pub use core::time::Duration;

#[unstable(feature = "duration_checked_float", issue = "83400")]
pub use core::time::FromSecsError;

/// A measurement of a monotonically nondecreasing clock.
/// Opaque and useful only with [`Duration`].
///
/// Instants are always guaranteed to be no less than any previously measured
/// instant when created, and are often useful for tasks such as measuring
/// benchmarks or timing how long an operation takes.
///
/// Note, however, that instants are not guaranteed to be **steady**. In other
/// words, each tick of the underlying clock might not be the same length (e.g.
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
///
/// # OS-specific behaviors
///
/// An `Instant` is a wrapper around system-specific types and it may behave
/// differently depending on the underlying operating system. For example,
/// the following snippet is fine on Linux but panics on macOS:
///
/// ```no_run
/// use std::time::{Instant, Duration};
///
/// let now = Instant::now();
/// let max_nanoseconds = u64::MAX / 1_000_000_000;
/// let duration = Duration::new(max_nanoseconds, 0);
/// println!("{:?}", now + duration);
/// ```
///
/// # Underlying System calls
/// Currently, the following system calls are being used to get the current time using `now()`:
///
/// |  Platform |               System call                                            |
/// |-----------|----------------------------------------------------------------------|
/// | SGX       | [`insecure_time` usercall]. More information on [timekeeping in SGX] |
/// | UNIX      | [clock_gettime (Monotonic Clock)]                                    |
/// | Darwin    | [mach_absolute_time]                                                 |
/// | VXWorks   | [clock_gettime (Monotonic Clock)]                                    |
/// | SOLID     | `get_tim`                                                            |
/// | WASI      | [__wasi_clock_time_get (Monotonic Clock)]                            |
/// | Windows   | [QueryPerformanceCounter]                                            |
///
/// [QueryPerformanceCounter]: https://docs.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter
/// [`insecure_time` usercall]: https://edp.fortanix.com/docs/api/fortanix_sgx_abi/struct.Usercalls.html#method.insecure_time
/// [timekeeping in SGX]: https://edp.fortanix.com/docs/concepts/rust-std/#codestdtimecode
/// [__wasi_clock_time_get (Monotonic Clock)]: https://github.com/WebAssembly/WASI/blob/master/phases/snapshot/docs.md#clock_time_get
/// [clock_gettime (Monotonic Clock)]: https://linux.die.net/man/3/clock_gettime
/// [mach_absolute_time]: https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/services/services.html
///
/// **Disclaimer:** These system calls might change over time.
///
/// > Note: mathematical operations like [`add`] may panic if the underlying
/// > structure cannot represent the new point in time.
///
/// [`add`]: Instant::add
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
///
/// # Underlying System calls
/// Currently, the following system calls are being used to get the current time using `now()`:
///
/// |  Platform |               System call                                            |
/// |-----------|----------------------------------------------------------------------|
/// | SGX       | [`insecure_time` usercall]. More information on [timekeeping in SGX] |
/// | UNIX      | [clock_gettime (Realtime Clock)]                                     |
/// | Darwin    | [gettimeofday]                                                       |
/// | VXWorks   | [clock_gettime (Realtime Clock)]                                     |
/// | SOLID     | `SOLID_RTC_ReadTime`                                                 |
/// | WASI      | [__wasi_clock_time_get (Realtime Clock)]                             |
/// | Windows   | [GetSystemTimePreciseAsFileTime] / [GetSystemTimeAsFileTime]         |
///
/// [`insecure_time` usercall]: https://edp.fortanix.com/docs/api/fortanix_sgx_abi/struct.Usercalls.html#method.insecure_time
/// [timekeeping in SGX]: https://edp.fortanix.com/docs/concepts/rust-std/#codestdtimecode
/// [gettimeofday]: https://man7.org/linux/man-pages/man2/gettimeofday.2.html
/// [clock_gettime (Realtime Clock)]: https://linux.die.net/man/3/clock_gettime
/// [__wasi_clock_time_get (Realtime Clock)]: https://github.com/WebAssembly/WASI/blob/master/phases/snapshot/docs.md#clock_time_get
/// [GetSystemTimePreciseAsFileTime]: https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimepreciseasfiletime
/// [GetSystemTimeAsFileTime]: https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimeasfiletime
///
/// **Disclaimer:** These system calls might change over time.
///
/// > Note: mathematical operations like [`add`] may panic if the underlying
/// > structure cannot represent the new point in time.
///
/// [`add`]: SystemTime::add
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
        // It seems that this just happens a lot in the wild.
        // We're seeing panics across various platforms where consecutive calls
        // to `Instant::now`, such as via the `elapsed` function, are panicking
        // as they're going backwards. Placed here is a last-ditch effort to try
        // to fix things up. We keep a global "latest now" instance which is
        // returned instead of what the OS says if the OS goes backwards.
        //
        // To hopefully mitigate the impact of this, a few platforms are
        // excluded as "these at least haven't gone backwards yet".
        if time::Instant::actually_monotonic() {
            return Instant(os_now);
        }

        Instant(monotonic::monotonize(os_now))
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
    /// or None if that instant is later than this one.
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
    /// println!("{:?}", new_now.checked_duration_since(now));
    /// println!("{:?}", now.checked_duration_since(new_now)); // None
    /// ```
    #[stable(feature = "checked_duration_since", since = "1.39.0")]
    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.0.checked_sub_instant(&earlier.0)
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
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
    /// println!("{:?}", new_now.saturating_duration_since(now));
    /// println!("{:?}", now.saturating_duration_since(new_now)); // 0ns
    /// ```
    #[stable(feature = "checked_duration_since", since = "1.39.0")]
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
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
    /// underlying data structure. See [`Instant::checked_add`] for a version without panic.
    fn add(self, other: Duration) -> Instant {
        self.checked_add(other).expect("overflow when adding duration to instant")
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
        self.checked_sub(other).expect("overflow when subtracting duration from instant")
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
    /// [`Instant`] can be used to measure elapsed time without this risk of failure.
    ///
    /// If successful, <code>[Ok]\([Duration])</code> is returned where the duration represents
    /// the amount of time elapsed from the specified measurement to this one.
    ///
    /// Returns an [`Err`] if `earlier` is later than `self`, and the error
    /// contains how far from `self` the time is.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::SystemTime;
    ///
    /// let sys_time = SystemTime::now();
    /// let new_sys_time = SystemTime::now();
    /// let difference = new_sys_time.duration_since(sys_time)
    ///     .expect("Clock may have gone backwards");
    /// println!("{:?}", difference);
    /// ```
    #[stable(feature = "time2", since = "1.8.0")]
    pub fn duration_since(&self, earlier: SystemTime) -> Result<Duration, SystemTimeError> {
        self.0.sub_time(&earlier.0).map_err(SystemTimeError)
    }

    /// Returns the difference between the clock time when this
    /// system time was created, and the current clock time.
    ///
    /// This function may fail as the underlying system clock is susceptible to
    /// drift and updates (e.g., the system clock could go backwards), so this
    /// function might not always succeed. If successful, <code>[Ok]\([Duration])</code> is
    /// returned where the duration represents the amount of time elapsed from
    /// this time measurement to the current time.
    ///
    /// To measure elapsed time reliably, use [`Instant`] instead.
    ///
    /// Returns an [`Err`] if `self` is later than the current system time, and
    /// the error contains how far from the current system time `self` is.
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
    /// underlying data structure. See [`SystemTime::checked_add`] for a version without panic.
    fn add(self, dur: Duration) -> SystemTime {
        self.checked_add(dur).expect("overflow when adding duration to instant")
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
        self.checked_sub(dur).expect("overflow when subtracting duration from instant")
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
    /// A `SystemTimeError` is returned from the [`SystemTime::duration_since`]
    /// and [`SystemTime::elapsed`] methods whenever the second system time
    /// represents a point later in time than the `self` of the method call.
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
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "other time was not earlier than self"
    }
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
