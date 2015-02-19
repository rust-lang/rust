- Feature Name: none?
- Start Date: 2015-02-18
- RFC PR: [rust-lang/rfcs#886](https://github.com/rust-lang/rfcs/pull/886)
- Rust Issue: (leave this empty)

# Summary

Support the `#[must_use]` attribute on arbitrary functions, to make
the compiler lint when a call to such a function is ignored. Mark
`Result::{ok, err}` `#[must_use]`.

# Motivation

The `#[must_use]` lint is extremely useful for ensuring that values
that are likely to be important are handled, even if by just
explicitly ignoring them with, e.g., `let _ = ...;`. This expresses
the programmers intention clearly, so that there is less confusion
about whether, for example, ignoring the possible error from a `write`
call is intentional or just an accidental oversight.

Rust has got a lot of mileage out connecting the `#[must_use]` lint to
specific types: types like `Result`, `MutexGuard` (any guard, ina
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

However, not every "important" (or, "usually want to use") result can
be a type that can be marked `#[must_use]`, for example, sometimes
functions return unopinionated type like `Option<...>` or `u8` that
may lead to confusion if they are ignored. For example, the `Result<T,
E>` type provides

```rust
pub fn ok(self) -> Option<T> {
    match self {
        Ok(x)  => Some(x),
        Err(_) => None,
    }
}
```

to view any data in the `Ok` variant as an `Option`. Notably, this
does no meaningful computation, in particular, it does not *enforce*
that the `Result` is `ok()`. Someone reading a line of code
`returns_result().ok();` where the returned value is unused
cannot easily tell if that behaviour was correct, or if something else
was intended, possibilities include:

- `let _ = returns_result();` to ignore the result (as
  `returns_result().ok();` does),
- `returns_result().unwrap();` to panic if there was an error,
- `returns_result().ok().something_else();` to do more computation.

This is somewhat problematic in the context of `Result` in particular,
because `.ok()` does not really (in the authors opinion) represent a
meaningful use of the `Result`, but it still silences the
`#[must_use]` error.

These cases can be addressed by allowing specific functions to
explicitly opt-in to also having important results, e.g. `#[must_use]
fn ok(self) -> Option<T>`. This is a natural generalisation of
`#[must_use]` to allow fine-grained control of context sensitive info.

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


# Drawbacks

This adds a little more complexity to the `#[must_use]` system.

The rule stated doesn't cover every instance where a `#[must_use]`
function is ignored, e.g. `(foo());` and `{ ...; foo() };` will not be
picked up, even though it is passing the result through a piece of
no-op syntax. This could be tweaked. Notably, the type-based rule doesn't
have this problem, since that sort of "passing-through" causes the
outer piece of syntax to be of the `#[must_use]` type, and so is
considered for the lint itself.

`Result::ok` is occasionally used for silencing the `#[must_use]`
error of `Result`, i.e. the ignoring of `foo().ok();` is
intentional. However, the most common way do ignore such things is
with `let _ =`, and `ok()` is rarely used in comparison, in most
code-bases: 2 instances in the rust-lang/rust codebase (vs. nearly 400
text matches for `let _ =`) and 4 in the servo/servo (vs. 55 `let _
=`). Yet another way to write this is `drop(foo())`, although neither
this nor `let _ =` have the method chaining style.

# Alternatives

- Adjust the rule to propagate `#[must_used]`ness through parentheses
  and blocks, so that `(foo());`, `{ foo() };` and even `if cond {
  foo() } else { 0 };` are linted.

- Provide an additional method on `Result`, e.g. `fn ignore(self) {}`, so
  that users who wish to ignore `Result`s can do so in the method
  chaining style: `foo().ignore();`.

# Unresolved questions

- Are there many other functions in the standard library/compiler
  would benefit from `#[must_use]`?
- Should this be feature gated?
