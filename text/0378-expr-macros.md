- Start Date: 2014-10-09
- RFC PR #: https://github.com/rust-lang/rfcs/pull/378
- Rust Issue #: https://github.com/rust-lang/rust/issues/18635

Summary
=======

Parse macro invocations with parentheses or square brackets as expressions no
matter the context, and require curly braces or a semicolon following the
invocation to invoke a macro as a statement.

Motivation
==========

Currently, macros that start a statement want to be a whole statement, and so
expressions such as `foo!().bar` don’t parse if they start a statement. The
reason for this is because sometimes one wants a macro that expands to an item
or statement (for example, `macro_rules!`), and forcing the user to add a
semicolon to the end is annoying and easy to forget for long, multi-line
statements. However, the vast majority of macro invocations are not intended to
expand to an item or statement, leading to frustrating parser errors.

Unfortunately, this is not as easy to resolve as simply checking for an infix
operator after every statement-like macro invocation, because there exist
operators that are both infix and prefix. For example, consider the following
function:

```rust
fn frob(x: int) -> int {
    maybe_return!(x)
    // Provide a default value
    -1
}
```

Today, this parses successfully. However, if a rule were added to the parser
that any macro invocation followed by an infix operator be parsed as a single
expression, this would still parse successfully, but not in the way expected: it
would be parsed as `(maybe_return!(x)) - 1`. This is an example of how it is
impossible to resolve this ambiguity properly without breaking compatibility.

Detailed design
===============

Treat all macro invocations with parentheses, `()`, or square brackets, `[]`, as
expressions, and never attempt to parse them as statements or items in a block
context unless they are followed directly by a semicolon. Require all
item-position macro invocations to be either invoked with curly braces, `{}`, or
be followed by a semicolon (for consistency).

This distinction between parentheses and curly braces has precedent in Rust:
tuple structs, which use parentheses, must be followed by a semicolon, while
structs with fields do not need to be followed by a semicolon. Many constructs
like `match` and `if`, which use curly braces, also do not require semicolons
when they begin a statement.

Drawbacks
=========

- This introduces a difference between different macro invocation delimiters,
  where previously there was no difference.
- This requires the use of semicolons in a few places where it was not necessary
  before.

Alternatives
============

- Require semicolons after all macro invocations that aren’t being used as
  expressions. This would have the downside of requiring semicolons after every
  `macro_rules!` declaration.

Unresolved questions
====================

None.
