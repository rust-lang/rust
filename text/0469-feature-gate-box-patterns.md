- Start Date: 2014-11-17
- RFC PR: [rust-lang/rfcs#469](https://github.com/rust-lang/rfcs/pull/469)
- Rust Issue: [rust-lang/rust#21931](https://github.com/rust-lang/rust/issues/21931)

# Summary

Move `box` patterns behind a feature gate.

# Motivation

A recent RFC (https://github.com/rust-lang/rfcs/pull/462) proposed renaming `box` patterns to `deref`. The discussion that followed indicates that while the language community may be in favour of some sort of renaming, there is no significant consensus around any concrete proposal, including the original one or any that emerged from the discussion.

This RFC proposes moving `box` patterns behind a feature gate to postpone that discussion and decision to when it becomes more clear how `box` patterns should interact with types other than `Box`.

In addition, in the future `box` patterns are expected to be made more general by enabling them to destructure any type that implements one of the `Deref` family of traits. As such a generalisation may potentially lead to some currently valid programs being rejected due to the interaction with type inference or other language features, it is desirable that this particular feature stays feature gated until then.

# Detailed design

A feature gate `box_patterns` will be defined and all uses of the `box` pattern will require said gate to be enabled.

# Drawbacks

Some currently valid Rust programs will have to opt in to another feature gate.

# Alternatives

Pursue https://github.com/rust-lang/rfcs/pull/462 before 1.0 and stabilise `box patterns` without a feature gate.

Leave `box` patterns as-is without putting them behind a feature gate.

# Unresolved questions

None.
