- Start Date: 2015-01-20
- RFC PR: [rust-lang/rfcs#601](https://github.com/rust-lang/rfcs/pull/601/)
- Rust Issue: [rust-lang/rust#22141](https://github.com/rust-lang/rust/issues/22141)

# Summary

Rename the `be` reserved keyword to `become`.

# Motivation

A keyword needs to be reserved to support guaranteed tail calls in a backward-compatible way. Currently the keyword reserved for this purpose is `be`, but the `become` alternative was proposed in
the old [RFC](https://github.com/rust-lang/rfcs/pull/81) for guaranteed tail calls, which is now postponed and tracked in [PR#271](https://github.com/rust-lang/rfcs/issues/271).

Some advantages of the `become` keyword are:
 - it provides a clearer indication of its meaning ("this function becomes that function")
 - its syntax results in better code alignment (`become` is exactly as long as `return`)

The expected result is that users will be unable to use `become` as identifier, ensuring that it will be available for future language extensions.

This RFC is not about implementing tail call elimination, only on whether the `be` keyword should be replaced with `become`.

# Detailed design

Rename the `be` reserved word to `become`. This is a very simple find-and-replace.

# Drawbacks

Some code might be using `become` as an identifier.

# Alternatives

The main alternative is to do nothing, i.e. to keep the `be` keyword reserved for supporting guaranteed tail calls in a backward-compatible way. Using `become` as the keyword for tail calls would not be backward-compatible because it would introduce a new keyword, which might have been used in valid code.

Another option is to add the `become` keyword, without removing `be`. This would have the same drawbacks as the current proposal (might break existing code), but it would also guarantee that the `become` keyword is available in the future.

# Unresolved questions

