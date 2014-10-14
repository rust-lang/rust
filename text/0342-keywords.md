- Start Date: 2014-10-07
- RFC PR: https://github.com/rust-lang/rfcs/pull/342
- Rust Issue: https://github.com/rust-lang/rust/issues/17862

# Summary

Reserve `abstract`, `final`, and `override` as possible keywords.

# Motivation

We intend to add some mechanism to Rust to support more efficient inheritance
(see, e.g., RFC PRs #245 and #250, and this
[thread](http://discuss.rust-lang.org/t/summary-of-efficient-inheritance-rfcs/494/43)
on discuss). Although we have not decided how to do this, we do know that we
will. Any implementation is likely to make use of keywords `virtual` (already
used, to remain reserved), `abstract`, `final`, and `override`, so it makes
sense to reserve these now to make the eventual implementation as backwards
compatible as possible.

# Detailed design

Make `abstract`, `final`, and `override` reserved keywords.

# Drawbacks

Takes a few more words out of the possible vocabulary of Rust programmers.

# Alternatives

Don't do this and deal with it when we have an implementation. This would mean
bumping the language version, probably.

# Unresolved questions

N/A
