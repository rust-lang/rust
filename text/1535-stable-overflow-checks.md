- Feature Name: N/A
- Start Date: 2016-03-09
- RFC PR: [rust-lang/rfcs#1535](https://github.com/rust-lang/rfcs/pull/1535)
- Rust Issue: [rust-lang/rust#33134](https://github.com/rust-lang/rust/issues/33134)

# Summary
[summary]: #summary

Stabilize the `-C overflow-checks` command line argument.

# Motivation
[motivation]: #motivation

This is an easy way to turn on overflow checks in release builds
without otherwise turning on debug assertions, via the `-C
debug-assertions` flag. In stable Rust today you can't get one without
the other.

Users can use the `-C overflow-checks` flag from their Cargo
config to turn on overflow checks for an entire application.

This flag, which accepts values of 'yes'/'no', 'on'/'off', is being
renamed from `force-overflow-checks` because the `force` doesn't add
anything that the 'yes'/'no'

# Detailed design
[design]: #detailed-design

This is a stabilization RFC. The only steps will be to move
`force-overflow-checks` from `-Z` to `-C`, renaming it to
`overflow-checks`, and making it stable.

# Drawbacks
[drawbacks]: #drawbacks

It's another rather ad-hoc flag for modifying code generation.

Like other such flags, this applies to the entire code unit,
regardless of monomorphizations. This means that code generation for a
single function can be diferent based on which code unit its
instantiated in.

# Alternatives
[alternatives]: #alternatives

The flag could instead be tied to crates such that any time code from
that crate is inlined/monomorphized it turns on overflow checks.

We might also want a design that provides per-function control over
overflow checks.

# Unresolved questions
[unresolved]: #unresolved-questions

Cargo might also add a profile option like

```toml
[profile.dev]
overflow-checks = true
```

This may also be accomplished by Cargo's pending support for passing
arbitrary flags to rustc.

