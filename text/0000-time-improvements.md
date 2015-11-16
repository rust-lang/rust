- Feature Name: time_improvements
- Start Date: 2015-09-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC proposes several new types and associated APIs for working with times in Rust.
The primary new types are `Instant`, for working with time that is guaranteed to be
monotonic, and `SystemTime`, for working with times across processes on a single system
(usually internally represented as a number of seconds since an epoch).

# Motivations

The primary motivation of this RFC is to flesh out a larger set of APIs for
representing instants in time and durations of time.

For various reasons that this RFC will explore, APIs related to time are fairly
error-prone and have a number of caveats that programmers do not expect.

Rust APIs tend to expose more of these kinds of caveats through their APIs, in
order to help programmers become aware of and handle edge-cases. At the same
time, un-ergonomic APIs can work against that goal.

This RFC attempts to balance the desire to expose common footguns and help
programmers handle edge-cases with a desire to avoid creating so many hoops to
jump through that the useful caveats get ignored.

At a high level, this RFC covers two concepts related to time:

* Instants, moments in time
* Durations, an amount of time between two instants

We would like to be able to do some basic operations with these instants:

* Compare two instants
* Add a time period to an instant
* Subtract a time period from an instant
* Compare an instant to "now" to discover time elapsed

However, there are a number of problems that arise when trying to define these
types and operations.

First of all, with the exception of moments in time created using system APIs that
guarantee monotonicity (because they were created within a single process, or
created during since the last boot), moments in time are not monotonic.
A simple example of this is that if a program creates two files sequentially,
it cannot assume that the creation time of the second file is later than the
creation time of the first file.

This is because NTP (the network time protocol) can arbitrarily change the
system clock, and can even **rewind time**. This kind of time travel means that
the "system time-line" is not continuous and monotonic, which is something that
programmers very often forget when writing code involving machine times.

This design attempts to help programmers avoid some of the most egregious and
unexpected consequences of this kind of "time travel".

---

Leap seconds, which cannot be predicted, mean that it is impossible
to reliably add a number of seconds to a particular moment in time represented
as a human date and time ("1 million seconds from 2015-09-20 at midnight").

They also mean that seemingly simple concepts, like "1 minute", have caveats
depending on exactly how they are used. Caveats related to leap seconds
create real-world bugs, because of how unusual leap seconds are, and how
unlikely programmers are to consider "12:00:60" as a valid time.

Certain kinds of seemingly simple operations may not make sense in
all cases. For example, adding "1 year" to February 29, 2012 would produce
February 29, 2013, which is not a valid date. Adding "1 month" to August 31,
2015 would produce September 31, 2015, which is also not a valid date.

Certain human descriptions of durations, like "1 month and 35 days"
do not make sense, and human descriptions like "1 month and 5 days" have
ambiguous meaning when used in operations (do you add 1 month first and then
5 days or vice versa).


For these reasons, this RFC does not attempt to define a human duration with
fields for years, days or months. Such a duration would be difficult to use
in operations without hard-to-remember ordering rules.

For these reasons, this RFC does not propose APIs related to human concepts
dates and times. It is intentionally forwards-compatible with such
extensions.

---

Finally, many APIs that **take** a `Duration` can only do something useful with
positive values. For example, a timeout API would not know how to wait a
negative amount of time before timing out. Even discounting the possibility of
coding mistakes, the problem of system clock time travel means that programmers
often produce negative durations that they did not expect, and APIs that
liberally accept negative durations only propagate the error further.

As a result, this RFC makes a number of simplifying assumptions that can be
relaxed over time with additional types or through further RFCs:

It provides convenience methods for constructing Durations from larger units
of time (minutes, hours, days), but gives them names like
`Duration.from_standard_hour`. A standard hour is always 3600 seconds,
regardless of leap seconds.

It provides APIs that are expected to produce positive `Duration`s, and expects
that APIs like timeouts will accept positive `Durations` (which is currently
the case in Rust's standard library). These APIs help the programmer discover
the possibility of system clock time travel, and either handle the error explicitly,
or at least avoid propagating the problem into other APIs (by using `unwrap`).

It separates monotonic time (`Instant`) from time derived from the system
clock (`SystemTime`), which must account for the possibility of time travel.
This allows methods related to monotonic time to be uncaveated, while working
with the system clock has more methods that return `Result`s.

This RFC does not attempt to define a type for calendared DateTimes, nor does it
directly address time zones.

# Proposal

## Types

```rust
pub struct Instant {
  secs: u64,
  nanos: u32
}

pub struct SystemTime {
  secs: u64,
  nanos: u32
}

pub struct Duration {
  secs: u64,
  nanos: u32
}
```

### Instant

`Instant` is the simplest of the types representing moments in time. It
represents an opaque (non-serializable!) timestamp that is guaranteed to
be monotonic when compared to another `Instant`.

> In this context, monotonic means that a timestamp created later in real-world
> time will always be not less than a timestamp created earlier in real-world
> time.

The `Duration` type can be used in conjunction with `Instant`, and these
operations have none of the usual time-related caveats.

* Add a `Duration` to a `Instant`, producing a new `Instant`
* compare two `Instant`s to each other
* subtract a `Instant` from a later `Instant`, producing a `Duration`
* ask for an amount of time elapsed since a `Instant`, producing a `Duration`

Asking for an amount of time elapsed from a given `Instant` is a very common
operation that is guaranteed to produce a positive `Duration`. Asking for the
difference between an earlier and a later `Instant` also produces a positive
`Duration` when used correctly.

This design does not assume that negative `Duration`s are never useful, but
rather that the most common uses of `Duration` do not have a meaningful
use for negative values. Rather than require each API that takes a `Duration`
to produce an `Err` (or `panic!`) when receiving a negative value, this design
optimizes for the broadly useful positive `Duration`.

```rust
impl Instant {
  /// Returns an instant corresponding to "now".
  pub fn now() -> Instant;

  /// Panics if `earlier` is later than &self.
  /// Because Instant is monotonic, the only time that `earlier` should be
  /// a later time is a bug in your code.
  pub fn duration_from_earlier(&self, earlier: Instant) -> Duration;

  /// Panics if self is later than the current time (can happen if a Instant
  /// is produced synthetically)
  pub fn elapsed(&self) -> Duration;
}

impl Add<Duration> for Instant {
  type Output = Instant;
}

impl Sub<Duration> for Instant {
  type Output = Instant;
}

impl PartialEq for Instant;
impl Eq for Instant;
impl PartialOrd for Instant;
impl Ord for Instant;
```

For convenience, several new constructors are added to `Duration`. Because any
unit greater than seconds has caveats related to leap seconds, all of the
constructors take "standard" units. For example a "standard minute" is 60
seconds, while a "standard hour" is 3600 seconds.

The "standard" terminology comes from [JodaTime][joda-time-standard].

[joda-time-standard]: http://joda-time.sourceforge.net/apidocs/org/joda/time/Duration.html#standardDays(long)

```rust
impl Duration {
  /// a standard minute is 60 seconds
  /// panics if the number of minutes is larger than u64 seconds
  pub fn from_standard_minutes(minutes: u64) -> Duration;

  /// a standard hour is 60 standard minutes
  /// panics if the number of hours is larger than u64 seconds
  pub fn from_standard_hours(hours: u64) -> Duration;

  /// a standard day is 24 standard hours
  /// panics if the number of days is larger than u64 seconds
  pub fn from_standard_days(days: u64) -> Duration;
}
```

### SystemTime

**This type should not be used for in-process timestamps, like those used in
benchmarks.**

A `SystemTime` represents a time stored on the local machine derived from the
system clock (in UTC). For example, it is used to represent `mtime` on the file
system.

The most important caveat of `SystemTime` is that it is **not monotonic**. This
means that you can save a file to the file system, then save another file to
the file system, **and the second file has an `mtime` earlier than the second**.

> **This means that an operation that happens after another operation in real
> time may have an earlier `SystemTime`!**

In practice, most programmers do not think about this kind of "time travel"
with the system clock, leading to strange bugs once the mistaken assumption
propagates through the system.

This design attempts to help the programmer catch the most egregious of these
kinds of mistakes (unexpected travel **back in time**) before the mistake
propagates.

```rust
impl SystemTime {
  /// Returns the system time corresponding to "now".
  pub fn now() -> SystemTime;

  /// Returns an `Err` if `earlier` is later
  pub fn duration_from_earlier(&self, earlier: SystemTime) -> Result<Duration, SystemTimeError>;

  /// Returns an `Err` if &self is later than the current system time.
  pub fn elapsed(&self) -> Result<Duration, SystemTimeError>;
}

impl Add<Duration> for SystemTime {
  type Output = SystemTime;
}

impl Sub<Duration> for SystemTime {
  type Output = SystemTime;
}

// An anchor which can be used to generate new SystemTime instances from a known
// Duration or convert a SystemTime to a Duration which can later then be used
// again to recreate the SystemTime.
//
// Defined to be "1970-01-01 00:00:00 UTC" on all systems.
const UNIX_EPOCH: SystemTime = ...;

// Note that none of these operations actually imply that the underlying system
// operation that produced these SystemTimes happened at the same time
// (for Eq) or before/after (for Ord) than the other system operation.
impl PartialEq for SystemTime;
impl Eq for SystemTime;
impl PartialOrd for SystemTime;
impl Ord for SystemTime;

impl SystemTimeError {
    /// A SystemTimeError originates from attempting to subtract two SystemTime
    /// instances, a and b. If a < b then an error is returned, and the duration
    /// returned represents (b - a).
    pub fn duration(&self) -> Duration;
}
```

The main difference from the design of `Instant` is that it is impossible to
know for sure that a `SystemTime` is in the past, even if the operation that
produced it happened in the past (in real time).

---

##### Illustrative Example:

If a program requests a `SystemTime` that represents the `mtime` of a given file,
then writes a new file and requests its `SystemTime`, it may expect the second
`SystemTime` to be after the first.

Using `duration_from_earlier` will remind the programmer that "time travel" is
possible, and make it easy to handle that case. As always, the programmer can
use `.unwrap()` in the prototype stage to avoid having to handle the edge-case
yet, while retaining a reminder that the edge-case is possible.

# Drawbacks

This RFC defines two new types for describing times, and posits a third type
to complete the picture. At first glance, having three different APIs for
working with times may seem overly complex.

However, there are significant differences between times that only go forward
and times that can go forward or backward. There are also significant differences
between times represented as a number since an epoch and time represented in
human terms.

As a result, this RFC chose to make these differences explicit, allowing
ergonomic, uncaveated use of monotonic time, and a small speedbump when
working with times that can move both forward and backward.

# Alternatives

One alternative design would be to attempt to have a single unified time
type. The rationale for not doing so is explained under Drawbacks.

Another possible alternative is to allow free math between instants,
rather than providing operations for comparing later instants to earlier
ones.

In practice, the vast majority of APIs **taking** a `Duration` expect
a positive-only `Duration`, and therefore code that subtracts a time
from another time will usually want a positive `Duration`.

The problem is especially acute when working with `SystemTime`, where
it is possible for a question like: "how much time has elapsed since
I created this file" to return a negative Duration!

This RFC attempts to catch mistakes related to negative `Duration`s at
the point where they are produced, rather than requiring all APIs that
**take** a `Duration` to guard against negative values.

Because `Ord` is implemented on `SystemTime` and `Instant`, it is
possible to compare two arbitrary times to each other first, and then
use `duration_from_earlier` reliably to get a positive `Duration`.

# Unresolved Questions

This RFC leaves types related to human representations of dates and times
to a future proposal.
