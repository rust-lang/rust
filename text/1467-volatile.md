- Feature Name: volatile
- Start Date: 2016-01-18
- RFC PR: [rust-lang/rfcs#1467](https://github.com/rust-lang/rfcs/pull/1467)
- Rust Issue: [rust-lang/rust#31756](https://github.com/rust-lang/rust/issues/31756)

# Summary
[summary]: #summary

Stabilize the `volatile_load` and `volatile_store` intrinsics as `ptr::read_volatile` and `ptr::write_volatile`.

# Motivation
[motivation]: #motivation

This is necessary to allow volatile access to memory-mapping I/O in stable code. Currently this is only possible using unstable intrinsics, or by abusing a bug in the `load` and `store` functions on atomic types which gives them volatile semantics ([rust-lang/rust#30962](https://github.com/rust-lang/rust/pull/30962)).

# Detailed design
[design]: #detailed-design

`ptr::read_volatile` and `ptr::write_volatile` will work the same way as `ptr::read` and `ptr::write` respectively, except that the memory access will be done with volatile semantics. The semantics of a volatile access are already pretty well defined by the C standard and by LLVM. In documentation we can refer to http://llvm.org/docs/LangRef.html#volatile-memory-accesses.

# Drawbacks
[drawbacks]: #drawbacks

None.

# Alternatives
[alternatives]: #alternatives

We could also stabilize the `volatile_set_memory`, `volatile_copy_memory` and `volatile_copy_nonoverlapping_memory` intrinsics as `ptr::write_bytes_volatile`, `ptr::copy_volatile` and `ptr::copy_nonoverlapping_volatile`, but these are not as widely used and are not available in C.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
