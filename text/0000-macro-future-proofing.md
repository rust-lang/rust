- Start Date: 2014-12-21
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

**NOTE**: Draft, not finalized.

# Key Terminology

- `macro`: anything invokable as `foo!(...)` in source code.
- `syntax extension`: a plugin to `rustc` that can provide macros or special
  handling for certain attributes.
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
    (start $foo:expr end) => ($foo),
}
```

`(start $foo:expr end)` is a matcher, `$foo` is an NT with `expr` as its
fragment specifier.

# Summary

Future-proof the allowed forms that input to an MBE can take by requiring
certain delimiters following NTs in a matcher.

# Motivation

In current Rust, the `macro_rules` parser is very liberal in what it accepts
in a matcher. This can cause problems, because it is possible to write an
MBE which corresponds to an ambiguous grammar. When an MBE is invoked, if the
macro parser encounters an amibuity while parsing, it will bail out with a
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

This is the bulk of the RFC. Explain the design in enough detail for somebody familiar
with the language to understand, and for somebody familiar with the compiler to implement.
This should get into specifics and corner-cases, and include examples of how the feature is used.

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
