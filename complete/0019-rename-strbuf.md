- Start Date: 2014-04-30
- RFC PR: [rust-lang/rfcs#60](https://github.com/rust-lang/rfcs/pull/60)
- Rust Issue: [rust-lang/rust#14312](https://github.com/rust-lang/rust/issues/14312)

# Summary

`StrBuf` should be renamed to `String`.

# Motivation

Since `StrBuf` is so common, it would benefit from a more traditional name.

# Drawbacks

It may be that `StrBuf` is a better name because it mirrors Java `StringBuilder` or C# `StringBuffer`. It may also be that `String` is confusing because of its similarity to `&str`.

# Detailed design

Rename `StrBuf` to `String`.

# Alternatives

The impact of not doing this would be that `StrBuf` would remain `StrBuf`.

# Unresolved questions

None.
