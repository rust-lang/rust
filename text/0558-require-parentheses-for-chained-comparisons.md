- Start Date: 2015-01-07
- RFC PR: [rust-lang/rfcs#558](https://github.com/rust-lang/rfcs/pull/558)
- Rust Issue: [rust-lang/rust#20724](https://github.com/rust-lang/rust/issues/20724)

# Summary

Remove chaining of comparison operators (e.g. `a == b == c`) from the syntax.
Instead, require extra parentheses (`(a == b) == c`).

# Motivation

```rust
fn f(a: bool, b: bool, c: bool) -> bool {
    a == b == c
}
```

This code is currently accepted and is evaluated as `((a == b) == c)`.
This may be confusing to programmers coming from languages like Python,
where chained comparison operators are evaluated as `(a == b && b == c)`.

In C, the same problem exists (and is excerbated by implicit conversions).
Styleguides like Misra-C require the use of parentheses in this case.

By requiring the use of parentheses, we avoid potential confusion now,
and open up the possibility for python-like chained comparisons post-1.0.

Additionally, making the chain `f < b > (c)` invalid allows us to easily produce
a diagnostic message: "Use `::<` instead of `<` if you meant to specify type arguments.",
which would be a vast improvement over the current diagnostics for this mistake.

# Detailed design

Emit a syntax error when a comparison operator appears as an operand of another comparison operator
(without being surrounded by parentheses).
The comparison operators are `<` `>` `<=` `>=` `==` and `!=`.

This is easily implemented directly in the parser.

Note that this restriction on accepted syntax will effectively merge the precedence level 4 (`<` `>` `<=` `>=`) with level 3 (`==` `!=`).

# Drawbacks

It's a breaking change.

In particular, code that currently uses the difference between precedence level 3 and 4 breaks
and will require the use of parentheses:

```rust
if a < 0 == b < 0 { /* both negative or both non-negative */ }
```

However, I don't think this kind of code sees much use.
The rustc codebase doesn't seem to have any occurrences of chained comparisons.

# Alternatives

As this RFC just makes the chained comparison syntax available for post-1.0 language features,
pretty much every alternative (including returning to the status quo) can still be implemented later.

If this RFC is not accepted, it will be impossible to add python-style chained comparison operators later.

A variation on this RFC would be to keep the separation between precedence level 3 and 4, and only reject programs
where a comparison operator appears as an operand of another comparison operator of the same precedence level.
This minimizes the breaking changes, but does not allow full python-style chained comparison operators in the future
(although a more limited form of them would still be possible).

# Unresolved questions

Is there real code that would get broken by this change?
So far, I've been unable to find any.
