- Start Date: 2014-06-15
- RFC PR #: [rust-lang/rfcs#123](https://github.com/rust-lang/rfcs/pull/123)
- Rust Issue #: [rust-lang/rust#16281](https://github.com/rust-lang/rust/issues/16281)

# Summary

Rename the `Share` trait to `Sync`

# Motivation

With interior mutability, the name "immutable pointer" for a value of type `&T`
is not quite accurate. Instead, the term "shared reference" is becoming popular
to reference values of type `&T`. The usage of the term "shared" is in conflict
with the `Share` trait, which is intended for types which can be safely shared
concurrently with a shared reference.

# Detailed design

Rename the `Share` trait in `std::kinds` to `Sync`. Documentation would
refer to `&T` as a shared reference and the notion of "shared" would simply mean
"many references" while `Sync` implies that it is safe to share among many
threads.

# Drawbacks

The name `Sync` may invoke conceptions of "synchronized" from languages such as
Java where locks are used, rather than meaning "safe to access in a shared
fashion across tasks".

# Alternatives

As any bikeshed, there are a number of other names which could be possible for
this trait:

* `Concurrent`
* `Synchronized`
* `Threadsafe`
* `Parallel`
* `Threaded`
* `Atomic`
* `DataRaceFree`
* `ConcurrentlySharable`

# Unresolved questions

None.
