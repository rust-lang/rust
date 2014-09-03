- Start Date: 2014-08-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/202
- Rust Issue: https://github.com/rust-lang/rust/issues/16967

# Summary

Change syntax of subslices matching from `..xs` to `xs..`
to be more consistent with the rest of the language
and allow future backwards compatible improvements.

Small example:

```rust
match slice {
    [xs.., _] => xs,
    [] => fail!()
}
```

This is basically heavily stripped version of [RFC 101](https://github.com/rust-lang/rfcs/pull/101).

# Motivation

In Rust, symbol after `..` token usually describes number of things,
as in `[T, ..N]` type or in `[e, ..N]` expression.
But in following pattern: `[_, ..xs]`, `xs` doesn't describe any number,
but the whole subslice.

I propose to move dots to the right for several reasons (including one mentioned above):

1. Looks more natural (but that might be subjective).
2. Consistent with the rest of the language.
3. C++ uses `args...` in variadic templates.
4. It allows extending slice pattern matching as described in [RFC 101](https://github.com/rust-lang/rfcs/pull/101).

# Detailed design

Slice matching grammar would change to (assuming trailing commas;
grammar syntax as in Rust manual):

    slice_pattern : "[" [[pattern | subslice_pattern] ","]* "]" ;
    subslice_pattern : ["mut"? ident]? ".." ["@" slice_pattern]? ;

To compare, currently it looks like:

    slice_pattern : "[" [[pattern | subslice_pattern] ","]* "]" ;
    subslice_pattern : ".." ["mut"? ident ["@" slice_pattern]?]? ;

# Drawbacks

Backward incompatible.

# Alternatives

Don't do it at all.

# Unresolved questions

Whether subslice matching combined with `@` should be written as `xs.. @[1, 2]`
or maybe in another way: `xs @[1, 2]..`.
