- Start Date: 2014-07-14
- RFC PR #: [rust-lang/rfcs#164](https://github.com/rust-lang/rfcs/pull/164)
- Rust Issue #: [rust-lang/rust#16951](https://github.com/rust-lang/rust/issues/16951)

# Summary

Rust's support for pattern matching on slices has grown steadily and incrementally without a lot of oversight.
We have concern that Rust is doing too much here, and that the complexity is not worth it. This RFC proposes
to feature gate multiple-element slice matches in the head and middle positions (`[xs.., 0, 0]` and `[0, xs.., 0]`).

# Motivation

Some general reasons and one specific: first, the implementation of Rust's match machinery is notoriously complex, and not well-loved. Removing features is seen as a valid way to reduce complexity. Second, slice matching in particular, is difficult to implement, while also being of only moderate utility (there are many types of collections - slices just happen to be built into the language). Finally, the exhaustiveness check is not correct for slice patterns because of their complexity; it's not known if it
can be done correctly, nor whether it is worth the effort to do so.

# Detailed design

The `advanced_slice_patterns` feature gate will be added. When the compiler encounters slice pattern matches in head or middle position it will emit a warning or error according to the current settings.

# Drawbacks

It removes two features that some people like.

# Alternatives

Fixing the exhaustiveness check would allow the feature to remain.

# Unresolved questions

N/A
