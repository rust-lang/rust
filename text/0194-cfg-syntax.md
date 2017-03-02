- Start Date: 2014-08-09
- RFC PR #: [rust-lang/rfcs#194](https://github.com/rust-lang/rfcs/pull/194)
- Rust Issue: [rust-lang/rust#17490](https://github.com/rust-lang/rust/issues/17490)

# Summary

The `#[cfg(...)]` attribute provides a mechanism for conditional compilation of
items in a Rust crate. This RFC proposes to change the syntax of `#[cfg]` to
make more sense as well as enable expansion of the conditional compilation
system to attributes while maintaining a single syntax.

# Motivation

In the current implementation, `#[cfg(...)]` takes a comma separated list of
`key`, `key = "value"`, `not(key)`, or `not(key = "value")`. An individual
`#[cfg(...)]` attribute "matches" if *all* of the contained cfg patterns match
the compilation environment, and an item preserved if it *either* has no
`#[cfg(...)]` attributes or *any* of the `#[cfg(...)]` attributes present
match.

This is problematic for several reasons:

* It is excessively verbose in certain situations. For example, implementing
    the equivalent of `(a AND (b OR c OR d))` requires three separate
    attributes and `a` to be duplicated in each.
* It differs from all other attributes in that all `#[cfg(...)]` attributes on
    an item must be processed together instead of in isolation. This change
    will move `#[cfg(...)]` closer to implementation as a normal syntax
    extension.

# Detailed design

The `<p>` inside of `#[cfg(<p>)]` will be called a *cfg pattern* and have a
simple recursive syntax:

* `key` is a cfg pattern and will match if `key` is present in the
    compilation environment.
* `key = "value"` is a cfg pattern and will match if a mapping from `key`
    to `value` is present in the compilation environment. At present, key-value
    pairs only exist for compiler defined keys such as `target_os` and
    `endian`.
* `not(<p>)` is a cfg pattern if `<p>` is and matches if `<p>` does not match.
* `all(<p>, ...)` is a cfg pattern if all of the comma-separated `<p>`s are cfg
    patterns and all of them match.
* `any(<p>, ...)` is a cfg pattern if all of the comma-separated `<p>`s are cfg
    patterns and any of them match.

If an item is tagged with `#[cfg(<p>)]`, that item will be stripped from the
AST if the cfg pattern `<p>` does not match.

One implementation hazard is that the semantics of
```rust
#[cfg(a)]
#[cfg(b)]
fn foo() {}
```
will change from "include `foo` if *either of* `a` and `b` are present in the
compilation environment" to "include `foo` if *both of* `a` and `b` are present
in the compilation environment". To ease the transition, the old semantics of
multiple `#[cfg(...)]` attributes will be maintained as a special case, with a
warning. After some reasonable period of time, the special case will be
removed.

In addition, `#[cfg(a, b, c)]` will be accepted with a warning and be
equivalent to `#[cfg(all(a, b, c))]`. Again, after some reasonable period of
time, this behavior will be removed as well.

The `cfg!()` syntax extension will be modified to accept cfg patterns as well.
A `#[cfg_attr(<p>, <attr>)]` syntax extension will be added
([PR 16230](https://github.com/rust-lang/rust/pull/16230)) which will expand to
`#[<attr>]` if the cfg pattern `<p>` matches.  The test harness's
`#[ignore]` attribute will have its built-in cfg filtering
functionality stripped in favor of `#[cfg_attr(<p>, ignore)]`.

# Drawbacks

While the implementation of this change in the compiler will be
straightforward, the effects on downstream code will be significant, especially
in the standard library.

# Alternatives

`all` and `any` could be renamed to `and` and `or`, though I feel that the
proposed names read better with the function-like syntax and are consistent
with `Iterator::all` and `Iterator::any`.

Issue [#2119](https://github.com/rust-lang/rust/issues/2119) proposed the
addition of `||` and `&&` operators and parantheses to the attribute syntax
to result in something like `#[cfg(a || (b && c)]`. I don't favor this proposal
since it would result in a major change to the attribute syntax for relatively
little readability gain.

# Unresolved questions

How long should multiple `#[cfg(...)]` attributes on a single item be
forbidden? It should probably be at least until after 0.12 releases.

Should we permanently keep the behavior of treating `#[cfg(a, b)]` as
`#[cfg(all(a, b))]`? It is the common case, and adding this interpretation
can reduce the noise level a bit. On the other hand, it may be a bit confusing
to read as it's not immediately clear if it will be processed as `and(..)` or
`all(..)`.
