- Feature Name: none?
- Start Date: 2015-02-18
- RFC PR: https://github.com/rust-lang/rfcs/pull/1940
- Rust Issue: https://github.com/rust-lang/rust/issues/43302

# Summary

Support the `#[must_use]` attribute on arbitrary functions, to make
the compiler lint when a call to such a function is ignored. Mark
`PartialEq::{eq, ne}` `#[must_use]` as well as `PartialOrd::{lt, gt, le, ge}`.

# Motivation

The `#[must_use]` lint is extremely useful for ensuring that values
that are likely to be important are handled, even if by just
explicitly ignoring them with, e.g., `let _ = ...;`. This expresses
the programmers intention clearly, so that there is less confusion
about whether, for example, ignoring the possible error from a `write`
call is intentional or just an accidental oversight.

Rust has got a lot of mileage out connecting the `#[must_use]` lint to
specific types: types like `Result`, `MutexGuard` (any guard, in
general) and the lazy iterator adapters have narrow enough use cases
that the programmer usually wants to do something with them. These
types are marked `#[must_use]` and the compiler will print an error if
a semicolon ever throws away a value of that type:

```rust
fn returns_result() -> Result<(), ()> {
    Ok(())
}

fn ignore_it() {
    returns_result();
}
```

```
test.rs:6:5: 6:11 warning: unused result which must be used, #[warn(unused_must_use)] on by default
test.rs:6     returns_result();
              ^~~~~~~~~~~~~~~~~
```

One of the most important use-cases for this would be annotating `PartialEq::{eq, ne}` with `#[must_use]`.

There's a bug in Android where instead of `modem_reset_flag = 0;` the file affected has `modem_reset_flag == 0;`.
Rust does not do better in this case. If you wrote `modem_reset_flag == false;` the compiler would be perfectly happy and wouldn't warn you. By marking `PartialEq` `#[must_use]` the compiler would complain about things like:

```
    modem_reset_flag == false; //warning
    modem_reset_flag = false; //ok
```

See further discussion in [#1812.](https://github.com/rust-lang/rfcs/pull/1812)

# Detailed design

If a semicolon discards the result of a function or method tagged with
`#[must_use]`, the compiler will emit a lint message (under same lint
as `#[must_use]` on types). An optional message `#[must_use = "..."]`
will be printed, to provide the user with more guidance.

```rust
#[must_use]
fn foo() -> u8 { 0 }


struct Bar;

impl Bar {
     #[must_use = "maybe you meant something else"]
     fn baz(&self) -> Option<String> { None }
}

fn qux() {
    foo(); // warning: unused result that must be used
    Bar.baz(); // warning: unused result that must be used: maybe you meant something else
}
```

The primary motivation is to mark `PartialEq` functions as `#[must_use]`:

```
#[must_use = "the result of testing for equality should not be discarded"]
fn eq(&self, other: &Rhs) -> bool;
```

The same thing for `ne`, and also `lt`, `gt`, `ge`, `le` in `PartialOrd`. There is no reason to discard the results of those operations. This means the `impl`s of these functions are not changed, it still issues a warning even for a custom `impl`.

# Drawbacks

This adds a little more complexity to the `#[must_use]` system, and
may be misused by library authors (but then, many features may be
misused).

The rule stated doesn't cover every instance where a `#[must_use]`
function is ignored, e.g. `(foo());` and `{ ...; foo() };` will not be
picked up, even though it is passing the result through a piece of
no-op syntax. This could be tweaked. Notably, the type-based rule doesn't
have this problem, since that sort of "passing-through" causes the
outer piece of syntax to be of the `#[must_use]` type, and so is
considered for the lint itself.

Marking functions `#[must_use]` is a breaking change in certain cases,
e.g. if someone is ignoring their result and has the relevant lint (or
warnings in general) set to be an error. This is a general problem of
improving/expanding lints.

# Alternatives

- Adjust the rule to propagate `#[must_used]`ness through parentheses
  and blocks, so that `(foo());`, `{ foo() };` and even `if cond {
  foo() } else { 0 };` are linted.
  
- Should we let particular `impl`s of a function have this attribute? Current design allows you to attach it inside the declaration of the trait.

# Unresolved questions

- Should this be feature gated?
