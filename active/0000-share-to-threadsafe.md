- Start Date: 2014-06-15
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Rename the `Share` trait to `Threadsafe`.

# Motivation

With interior mutability, the name "immutable pointer" for a value of type `&T`
is not quite accurate. Instead, the term "shared reference" is becoming popular
to reference values of type `&T`. The usage of the term "shared" is in conflict
with the `Share` trait, which is intended for types which can be safely shared
concurrently with a shared reference.

# Detailed design

Rename the `Share` trait in `std::kinds` to `Threadsafe`. Documentation would
refer to `&T` as a shared reference and the notion of "shared" would simply mean
"many references" while `Threadsafe` implies that it is safe to share among many
threads.

# Drawbacks

The name `Threadsafe` may imply that a type is itself always safe to use
concurrently. While it is impossible to safely have *data races*, it is possible
to safely have *race conditions*. The name `Threadsafe` may imply that a type
has no race conditions, which is not quite accurate.

Additionally, the name `Threadsafe` is 5 letters longer than `Share`, which is a
little unfortunate.

# Alternatives

As any bikeshed, there are a number of other names which could be possible for
this trait:

* `Concurrent`
* `Synchronized`
* `Parallel`
* `Threaded`
* `Atomic`
* `DataRaceFree`
* `ConcurrentlySharable`

# Unresolved questions

None.
