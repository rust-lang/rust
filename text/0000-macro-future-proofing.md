- Start Date: 2014-12-21
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Key Terminology

- `macro`: anything invokable as `foo!(...)` in source code.
- `MBE`: macro-by-example, a macro defined by `macro_rules`.
- `matcher`: the left-hand-side of a rule in a `macro_rules` invocation.
- `macro parser`: the bit of code in the Rust parser that will parse the input
  using a grammar derived from all of the matchers.
- `NT`: non-terminal, the various "meta-variables" that can appear in a matcher.
- `fragment`: The piece of Rust syntax that an NT can accept.
- `fragment specifier`: The identifier in an NT that specifies which fragment
  the NT accepts.
- `language`: a context-free language.

Example:

```rust
macro_rules! i_am_an_mbe {
    (start $foo:expr end) => ($foo)
}
```

`(start $foo:expr end)` is a matcher, `$foo` is an NT with `expr` as its
fragment specifier.

# Summary

Future-proof the allowed forms that input to an MBE can take by requiring
certain delimiters following NTs in a matcher. In the future, it will be
possible to lift these restrictions backwards compatibly if desired.

# Motivation

In current Rust, the `macro_rules` parser is very liberal in what it accepts
in a matcher. This can cause problems, because it is possible to write an
MBE which corresponds to an ambiguous grammar. When an MBE is invoked, if the
macro parser encounters an ambiguity while parsing, it will bail out with a
"local ambiguity" error. As an example for this, take the following MBE:

```rust
macro_rules! foo {
    ($($foo:expr)* $bar:block) => (/*...*/)
};
```

Attempts to invoke this MBE will never succeed, because the macro parser
will always emit an ambiguity error rather than make a choice when presented
an ambiguity. In particular, it needs to decide when to stop accepting
expressions for `foo` and look for a block for `bar` (noting that blocks are
valid expressions). Situations like this are inherent to the macro system. On
the other hand, it's possible to write an unambiguous matcher that becomes
ambiguous due to changes in the syntax for the various fragments. As a
concrete example:

```rust
macro_rules! bar {
    ($in:ty ( $($arg:ident)*, ) -> $out:ty;) => (/*...*/)
};
```

When the type syntax was extended to include the unboxed closure traits,
an input such as `FnMut(i8, u8) -> i8;` became ambiguous. The goal of this
proposal is to prevent such scenarios in the future by requiring certain
"delimiter tokens" after an NT. When extending Rust's syntax in the future,
ambiguity need only be considered when combined with these sets of delimiters,
rather than any possible arbitrary matcher.

# Detailed design

The algorithm for recognizing valid matchers `M` follows. Note that a matcher
is merely a token tree. A "simple NT" is an NT without repetitions. That is,
`$foo:ty` is a simple NT but `$($foo:ty)+` is not. `FOLLOW(NT)` is the set of
allowed tokens for the given NT's fragment specifier, and is defined below.
`F` is used for representing the separator in complex NTs.  In `$($foo:ty),+`,
`F` would be `,`, and for `$($foo:ty)+`, `F` would be `EOF`.

*input*: a token tree `M` representing a matcher and a token `F`

*output*: whether M is valid

1. If there are no tokens in `M`, accept.
2. For each token `T` in `M`:
    1. If `T` is not an NT, continue.
    2. If `T` is a simple NT, look ahead to the next token `T'` in `M`. If
       `T'` is `EOF` or a close delimiter of a token tree, replace `T'` with
       `F`. If `T'` is in the set `FOLLOW(NT)`, `T'` is EOF, `T'` is any NT,
       or `T'` is any identifier, continue. Else, reject.
    3. Else, `T` is a complex NT.
        1. If `T` has the form `$(...)+` or `$(...)*`, run the algorithm on
           the contents with `F` set to `EOF`. If it accepts, continue, else,
           reject.
        2. If `T` has the form `$(...)U+` or `$(...)U*` for some token `U`, run
           the algorithm on the contents with `F` set to `U`. If it accepts,
           continue, else, reject.

This algorithm should be run on every matcher in every `macro_rules`
invocation, with `F` as `EOF`. If it rejects a matcher, an error should be
emitted and compilation should not complete.

The current legal fragment specifiers are: `item`, `block`, `stmt`, `pat`,
`expr`, `ty`, `ident`, `path`, `meta`, and `tt`.

- `FOLLOW(stmt)` = `FOLLOW(expr)`
- `FOLLOW(pat)` = `{FatArrow, Comma, Pipe}`
- `FOLLOW(expr)` = `{Comma, FatArrow, CloseBrace, CloseParen, CloseBracket}`
- `FOLLOW(ty)` = `{Comma, CloseBrace, CloseParen, CloseBracket}`
- `FOLLOW(block)` = any token
- `FOLLOW(ident)` = any token
- `FOLLOW(tt)` = any token
- `FOLLOW(item)` = up for discussion
- `FOLLOW(path)` = up for discussion
- `FOLLOW(meta)` = up for discussion

# Drawbacks

It does restrict the input to a MBE, but the choice of delimiters provides
reasonable freedom.

# Alternatives

1. Fix the syntax that a fragment can parse. This would create a situation
   where a future MBE might not be able to accept certain inputs because the
   input uses newer features than the fragment that was fixed at 1.0. For
   example, in the `bar` MBE above, if the `ty` fragment was fixed before the
   unboxed closure sugar was introduced, the MBE would not be able to accept
   such a type. While this approach is feasible, it would cause unnecessary
   confusion for future users of MBEs when they can't put certain perfectly
   valid Rust code in the input to an MBE. Versioned fragments could avoid
   this problem but only for new code.
2. Keep `macro_rules` unstable. Given the great syntactical abstraction that
   `macro_rules` provides, it would be a shame for it to be unusable in a
   release version of Rust. If ever `macro_rules` were to be stabilized, this
   same issue would come up.
3. Do nothing. This is very dangerous, and has the potential to essentially
   freeze Rust's syntax for fear of accidentally breaking a macro.

# Unresolved questions

1. What should the FOLLOW sets for `item`, `path`, and `meta` be?
2. Should the `FOLLOW` set for `ty` be extended? In practice, `RArrow`,
   `Colon`, `as`, and `in` are also used. (See next item)
2. What, if any, identifiers should be allowed in the FOLLOW sets? The author
   is concerned that allowing arbitrary identifiers would limit the future use
   of "contextual keywords".
