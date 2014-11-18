- Start Date: 2014--28
- RFC PR: [#463](https://github.com/rust-lang/rfcs/pull/463)
- Rust Issue: [#19088](https://github.com/rust-lang/rust/issues/19088)

# Summary

Include identifiers immediately after literals in the literal token to
allow future expansion, e.g. `"foo"bar` and a `1baz` are considered
whole (but semantically invalid) tokens, rather than two separate
tokens `"foo"`, `bar` and `1`, `baz` respectively. This allows future
expansion of handling literals without risking breaking (macro) code.


# Motivation

Currently a few kinds of literals (integers and floats) can have a
fixed set of suffixes and other kinds do not include any suffixes. The
valid suffixes on numbers are:


```text
u, u8, u16, u32, u64
i, i8, i16, i32, i64
f32, f64
```

Most things not in this list are just ignored and treated as an
entirely separate token (prefixes of `128` are errors: e.g. `1u12` has
an error `"invalid int suffix"`), and similarly any suffixes on other
literals are also separate tokens. For example:

```rust
#![feature(macro_rules)]

// makes a tuple
macro_rules! foo( ($($a: expr)*) => { ($($a, )+) } )

fn main() {
    let bar = "suffix";
    let y = "suffix";

    let t: (uint, uint) = foo!(1u256);
    println!("{}", foo!("foo"bar));
    println!("{}", foo!('x'y));
}
/*
output:
(1, 256)
(foo, suffix)
(x, suffix)
*/
```

The compiler is eating the `1u` and then seeing the invalid suffix
`256` and so treating that as a separate token, and similarly for the
string and character literals. (This problem is only visible in
macros, since that is the only place where two literals/identifiers can be placed
directly adjacent.)

This behaviour means we would be unable to expand the possibilities
for literals after freezing the language/macros, which would be
unfortunate, since [user defined literals in C++][cpp] are reportedly
very nice, proposals for "bit data" would like to use types like `u1`
and `u5` (e.g. [RFC PR 327][327]), and there are "fringe" types like
[`f16`][f16], [`f128`][f128] and `u128` that have uses but are not
common enough to warrant adding to the language now.

[cpp]: http://en.cppreference.com/w/cpp/language/user_literal
[327]: https://github.com/rust-lang/rfcs/pull/327
[f16]: http://en.wikipedia.org/wiki/Half-precision_floating-point_format
[f128]: https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format

# Detailed design

The tokenizer will have grammar `literal: raw_literal identifier?`
where `raw_literal` covers strings, characters and numbers without
suffixes (e.g. `"foo"`, `'a'`, `1`, `0x10`).

Examples of "valid" literals after this change (that is, entities that
will be consumed as a single token):

```
"foo"bar "foo"_baz
'a'x 'a'_y

15u16 17i18 19f20 21.22f23
0b11u25 0x26i27 28.29e30f31

123foo 0.0bar
```

Placing a space between the letter of the suffix and the literal will
cause it to be parsed as two separate tokens, just like today. That is
`"foo"bar` is one token, `"foo" bar` is two tokens.

The example above would then be an error, something like:

```rust
    let t: (uint, uint) = foo!(1u256); // error: literal with unsupported size
    println!("{}", foo!("foo"bar)); // error: literal with unsupported suffix
    println!("{}", foo!('x'y)); // error: literal with unsupported suffix
```

The above demonstrates that numeric suffixes could be special cased
to detect `u<...>` and `i<...>` to give more useful error messages.

(The macro example there is definitely an error because it is using
the incorrectly-suffixed literals as `expr`s. If it was only
handling them as a token, i.e. `tt`, there is the possibility that it
wouldn't have to be illegal, e.g. `stringify!(1u256)` doesn't have to
be illegal because the `1u256` never occurs at runtime/in the type
system.)

# Drawbacks

None beyond outlawing placing a literal immediately before a pattern,
but the current behaviour can easily be restored with a space: `123u
456`. (If a macro is using this for the purpose of hacky generalised
literals, the unresolved question below touches on this.)

# Alternatives

Don't do this, or consider doing it for adjacent suffixes with an
alternative syntax, e.g. `10'bar` or `10$bar`.

# Unresolved questions

- Should it be the parser or the tokenizer rejecting invalid suffixes?
  This is effectively asking if it is legal for syntax extensions to
  be passed the raw literals? That is, can a `foo` procedural syntax
  extension accept and handle literals like `foo!(1u2)`?

- Should this apply to all expressions, e.g. `(1 + 2)bar`?
