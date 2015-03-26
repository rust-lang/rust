- Feature Name: Duration Reform
- Start Date: 2015-03-24
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC suggests stabilizing a reduced-scope `Duration` type that is appropriate for interoperating with various system calls that require timeouts. It does not stabilize a large number of conversion methods in `Duration` that have subtle caveats, with the intent of revisiting those conversions more holistically in the future.

# Motivation

There are a number of different notions of "time", each of which has a different set of caveats, and each of which can be designed for optimal ergonomics for its domain. This proposal focuses on one particular one: an amount of time in high-precision units.

Eventually, there are a number of concepts of time that deserve fleshed out APIs. Using the terminology from the popular Java time library [JodaTime][joda-time]:

* `Duration`: an amount of time, described in terms of a high
  precision unit.
* `Period`: an amount of time described in human terms ("5 minutes,
  27 seconds"), and which can only be resolved into a `Duration`
  relative to a moment in time.
* `Instant`: a moment in time represented in terms of a `Duration`
  since some epoch.

[joda-time]: http://www.joda.org/joda-time/

Human complications such as leap seconds, days in a month, and leap years, and machine complications such as NTP adjustments make these concepts and their full APIs more complicated than they would at first appear. This proposal focuses on fleshing out a design for `Duration` that is sufficient for use as a timeout, leaving the other concepts of time to a future proposal.

---

For the most part, the system APIs that this type is used to communicate with either use `timespec` (`u64` seconds plus `u32` nanos) or take a timeout in milliseconds (`u32` on Windows).

> For example, [`GetQueuedCompletionStatus`][iocp-ms-example], one of
> the primary APIs in the Windows IOCP API, takes a `dwMilliseconds`
> parameter as a [`DWORD`][msdn-dword], which is a `u32`. Some Windows
> APIs use "ticks" or 100-nanosecond units.

[iocp-ms-example]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa364986%28v=vs.85%29.aspx
[msdn-dword]: https://msdn.microsoft.com/en-us/library/cc230318.aspx

In light of that, this proposal has two primary goals:

* to define a type that can describe portable timeouts for cross-
  platform APIs
* to describe what should happen if a large `Duration` is passed into
  an API that does not accept timeouts that large

In general, this proposal considers it acceptable to reduce the granularity of timeouts (eliminating nanosecond granularity if only milliseconds are supported) and to truncate very large timeouts.

This proposal retains the two fields in the existing `Duration`:

* a `u64` of seconds
* a `u32` of additional nanosecond precision

Timeout APIs defined in terms of milliseconds will truncate `Duration`s that are more than `u32::MAX` in milliseconds, and will reduce the granularity of the nanosecond field.

> A `u32` of milliseconds supports a timeout longer than 45 days.

Future APIs to support a broader set of [Durations][joda-duration] APIs, a [Period][joda-period] and [Instant][joda-instant] type, as well as coercions between these types, would be useful, compatible follow-ups to this RFC.

[joda-duration]: http://www.joda.org/joda-time/key_duration.html
[joda-period]: http://www.joda.org/joda-time/key_period.html
[joda-instant]: http://www.joda.org/joda-time/key_instant.html

# Detailed design

A `Duration` represents a period of time represented in terms of nanosecond granularity. It has `u64` seconds and an additional `u32` nanoseconds. There is no concept of a negative `Duration`.

> A negative `Duration` has no meaning for many APIs that may wish
> to take a `Duration`, which means that all such APIs would need
> to decide what to do when confronted with a negative `Duration`.
> As a result, this proposal focuses on the predominant use-cases for
> `Duration`, where unsigned types remove a number of caveats and
> ambiguities.

```rust
pub struct Duration {
  secs: u64,
  nanos: u32 // may not be more than 1 billion
}

impl Duration {
    /// create a Duration from a number of seconds and an
    /// additional nanosecond precision
    pub fn new(secs: u64, nanos: u32) -> Timeout;

    /// create a Duration from a number of seconds
    pub fn from_secs(secs: u64) -> Timeout;

    /// create a Duration from a number of milliseconds
    pub fn from_millis(millis: u64) -> Timeout;

    /// the number of seconds represented by the Timeout
    pub fn secs(self) -> u64;

    /// the number of additional nanosecond precision
    pub fn nanos(self) -> u32;
}
```

When `Duration` is used with a system API that expects `u32` milliseconds, the nanosecond precision is dropped, and the time is truncated to `u32::MAX`.

`Duration` implements:

* `Add`, `Sub`, `Mul`, `Div` which follow the overflow and underflow
  rules for `u64` when applied to the `secs` field. Nanoseconds
  can never exceed 1 billion or be less than 0, and carry into the
  `secs` field.
* `Display`, which prints a number of seconds, milliseconds and
  nanoseconds (if more than 0).
* `Debug`, `Ord` (and `PartialOrd`), `Eq` (and `PartialEq`), `Copy`
  and `Clone`, which are derived.

This proposal does not, at this time, include mechanisms for instantiating a `Duration` from `weeks`, `days`, `hours` or `minutes`, because there are caveats to each of those units. In particular, the existence of leap seconds means that it is only possible to properly understand them relative to a particular starting point.

The Joda-Time library in Java explains the problem well [in their documentation][joda-period-confusion]:

[joda-period-confusion]: http://www.joda.org/joda-time/key_period.html

> A duration in Joda-Time represents a duration of time measured in milliseconds. The duration is often obtained from an interval. Durations are a very simple concept, and the implementation is also simple. They have no chronology or time zone, **and consist solely of the millisecond duration.**

> A period in Joda-Time represents a period of time defined in terms of fields, for example, 3 years 5 months 2 days and 7 hours. This differs from a duration in that it is inexact in terms of milliseconds. **A period can only be resolved to an exact number of milliseconds by specifying the instant (including chronology and time zone) it is relative to**.

In short, this is saying that people expect "23:50:00 + 10 minutes" to equal "00:00:00", but it's impossible to know for sure whether that's true unless you know the exact starting point so you can take leap seconds into consideration.

In order to address this confusion, Joda-Time's Duration has methods like `standardDays`/`toStandardDays` and `standardHours`/`toStandardHours`, which are meant to indicate to the user that the number of milliseconds is based on the standard number of milliseconds in an hour, rather than the colloquial notion of an "hour".

An approach like this could work for Rust, but this RFC is intentionally limited in scope to areas without substantial tradeoffs in an attempt to allow a minimal solution to progress more quickly.

This proposal does not include a method to get a number of milliseconds from a `Duration`, because the number of milliseconds could exceed `u64`, and we would have to decide whether to return an `Option`, panic, or wait for a standard bignum. In the interest of limiting this proposal to APIs with a straight-forward design, this proposal defers such a method.

# Drawbacks

The main drawback to this proposal is that it is significantly more minimal than the existing `Duration` API. However, this API is quite sufficient for timeouts, and without the caveats in the existing `Duration` API.

# Alternatives

We could stabilize the existing `Duration` API. However, it has a number of serious caveats:

* The caveats described above about some of the units it supports.
* It supports converting a `Duration` into a number of microseconds or
  nanoseconds. Because that cannot be done reliably, those methods
  return `Option`s, and APIs that need to convert `Duration` into
  nanoseconds have to re-surface the `Option` (unergonomic) or panic.
* More generally, it has a fairly large API surface area, and almost
  every method has some caveat that would need to be explored in order
  to stabilize it.

---

We could also include a number of convenience APIs that convert from other units into `Duration`s. This proposal assumes that some of those conveniences will eventually be added. However, the design of each of those conveniences is ambiguous, so they are not included in this initial proposal.

---

Finally, we could avoid any API for timeouts, and simply take milliseconds throughout the standard library. However, this has two drawbacks.

First, it does not allow us to represent higher-precision timeouts on systems that could support them.

Second, while this proposal does not yet include conveniences, it assumes that some conveniences should be added in the future once the design space is more fully explored. Starting with a simple type gives us space to grow into.

# Unresolved questions

* Should we implement all of the listed traits? Others?
