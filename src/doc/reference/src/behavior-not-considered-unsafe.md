## Behavior not considered unsafe

This is a list of behavior not considered *unsafe* in Rust terms, but that may
be undesired.

* Deadlocks
* Leaks of memory and other resources
* Exiting without calling destructors
* Integer overflow
  - Overflow is considered "unexpected" behavior and is always user-error,
    unless the `wrapping` primitives are used. In non-optimized builds, the compiler
    will insert debug checks that panic on overflow, but in optimized builds overflow
    instead results in wrapped values. See [RFC 560] for the rationale and more details.

[RFC 560]: https://github.com/rust-lang/rfcs/blob/master/text/0560-integer-overflow.md
