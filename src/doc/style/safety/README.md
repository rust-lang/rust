% Safety and guarantees

> **[FIXME]** Is there a better phrase than "strong guarantees" that encompasses
> both e.g. memory safety and e.g. data structure invariants?

A _guarantee_ is a property that holds no matter what client code does, unless
the client explicitly opts out:

* Rust guarantees memory safety and data-race freedom, with `unsafe`
  blocks as an opt-out mechanism.

* APIs in Rust often provide their own guarantees. For example, `std::str`
guarantees that its underlying buffer is valid utf-8. The `std::path::Path` type
guarantees no interior nulls. Both strings and paths provide `unsafe` mechanisms
for opting out of these guarantees (and thereby avoiding runtime checks).

Thinking about guarantees is an essential part of writing good Rust code.  The
rest of this subsection outlines some cross-cutting principles around
guarantees.
