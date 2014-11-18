- Start Date: 2014-11-02
- RFC PR: [rust-lang/rfcs#430](https://github.com/rust-lang/rfcs/pull/430)
- Rust Issue: [rust-lang/rust#19091](https://github.com/rust-lang/rust/issues/19091)

# Summary

This conventions RFC tweaks and finalizes a few long-running de facto
conventions, including capitalization/underscores, and the role of the `unwrap` method.

See [this RFC](https://github.com/rust-lang/rfcs/pull/328) for a competing proposal for `unwrap`.

# Motivation

This is part of the ongoing conventions formalization process. The
conventions described here have been loosely followed for a long time,
but this RFC seeks to nail down a few final details and make them
official.

# Detailed design

## General naming conventions

In general, Rust tends to use `CamelCase` for "type-level" constructs
(types and traits) and `snake_case` for "value-level" constructs. More
precisely, the proposed (and mostly followed) conventions are:

| Item | Convention |
| ---- | ---------- |
| Crates | `snake_case` (but prefer single word) |
| Modules | `snake_case` |
| Types | `CamelCase` |
| Traits | `CamelCase` |
| Enum variants | `CamelCase` |
| Functions | `snake_case` |
| Methods | `snake_case` |
| General constructors | `new` or `with_more_details` |
| Conversion constructors | `from_some_other_type` |
| Local variables | `snake_case` |
| Static variables | `SCREAMING_SNAKE_CASE` |
| Constant variables | `SCREAMING_SNAKE_CASE` |
| Type parameters | concise `CamelCase`, usually single uppercase letter: `T` |
| Lifetimes | short, lowercase: `'a` |

### Fine points

In `CamelCase`, acronyms count as one word: use `Uuid` rather than
`UUID`.  In `snake_case`, acronyms are lower-cased: `is_xid_start`.

In `snake_case` or `SCREAMING_SNAKE_CASE`, a "word" should never
consist of a single letter unless it is the last "word". So, we have
`btree_map` rather than `b_tree_map`, but `PI_2` rather than `PI2`.

## `unwrap`, `into_foo` and `into_inner`

There has been a [long](https://github.com/mozilla/rust/issues/13159)
[running](https://github.com/rust-lang/rust/pull/16436)
[debate](https://github.com/rust-lang/rust/pull/16436)
[about](https://github.com/rust-lang/rfcs/pull/328) the name of the
`unwrap` method found in `Option` and `Result`, but also a few other
standard library types. Part of the problem is that for some types
(e.g. `BufferedReader`), `unwrap` will never panic; but for `Option`
and `Result` calling `unwrap` is akin to asserting that the value is
`Some`/`Ok`.

There's basic agreement that we should have an unambiguous term for
the `Option`/`Result` version of `unwrap`. Proposals have included
`assert`, `ensure`, `expect`, `unwrap_or_panic` and others; see the
links above for extensive discussion. No clear consensus has emerged.

This RFC proposes a simple way out: continue to call the methods
`unwrap` for `Option` and `Result`, and rename *other* uses of
`unwrap` to follow conversion conventions. Whenever possible, these
panic-free unwrapping operations should be `into_foo` for some
concrete `foo`, but for generic types like `RefCell` the name
`into_inner` will suffice. By convention, these `into_` methods cannot
panic; and by (proposed) convention, `unwrap` should be reserved for
an `into_inner` conversion that *can*.

# Drawbacks

Not really applicable; we need to finalize these conventions.

# Unresolved questions

Are there remaining subtleties about the rules here that should be clarified?
