- Start Date: 2015-01-08
- RFC PR: [rust-lang/rfcs#565](https://github.com/rust-lang/rfcs/pull/565)
- Rust Issue: [rust-lang/rust#21436](https://github.com/rust-lang/rust/issues/21436)

# Summary

A [recent RFC](https://github.com/rust-lang/rfcs/pull/504) split what was
previously `fmt::Show` into two traits, `fmt::Show` and `fmt::String`, with
format specifiers `{:?}` and `{}` respectively.

That RFC did not, however, establish complete conventions for when to implement
which of the traits, nor what is expected from the output.  That's what this RFC
seeks to do.

It turns out that, due to the suggested conventions and other
concerns, renaming the traits is also desirable.

# Motivation

Part of the reason for splitting up `Show` in the first place was some tension
around the various use cases it was trying to cover, and the fact that it could
not cover them all simultaneously. Now that the trait has been split, this RFC
aims to provide clearer guidelines about their use.

# Detailed design

The design of the conventions stems from two basic desires:

1. It should be easy to generate a debugging representation of
   essentially any type.

2. It should be possible to create user-facing text output via convenient
   interpolation.

Part of the premise behind (2) is that user-facing output cannot automatically
be "composed" from smaller pieces of user-facing output (via, say,
`#[derive]`). Most of the time when you're preparing text for a user
consumption, the output needs to be quite tailored, and interpolation via
`format` is a good tool for that job.

As part of the conventions being laid out here, the RFC proposes to:

1. Rename `fmt::Show` to `fmt::Debug`, and
2. Rename `fmt::String` to `fmt::Display`.

## Debugging: `fmt::Debug`

The `fmt::Debug` trait is intended for debugging. It should:

* Be implemented on every type, usually via `#[derive(Debug)]`.
* Never panic.
* Escape away control characters.
* Introduce quotes and other delimiters as necessary to give a clear
  representation of the data involved.
* Focus on the *runtime* aspects of a type; repeating information such as
  suffixes for integer literals is not generally useful since that data is
  readily available from the type definition.

In terms of the output produced, the goal is make it easy to make sense of
compound data of various kinds without overwhelming debugging output
with every last bit of type information -- most of which is readily
available from the source. The following rules give rough guidance:

* Scalars print as unsuffixed literals.
* Strings print as normal quoted notation, with escapes.
* Smart pointers print as whatever they point to (without further annotation).
* Fully public structs print as you'd normally construct them:
  `MyStruct { f1: ..., f2: ... }`
* Enums print as you'd construct their variants (possibly with special
  cases for things like `Option` and single-variant enums?).
* Containers print using *some* notation that makes their type and
  contents clear. (Since we lack literals for all container types,
  this will be ad hoc).

It is *not* a *requirement* for the debugging output to be valid Rust
source. This is in general not possible in the presence of private
fields and other abstractions. However, when it is feasible to do so,
debugging output *should* match Rust syntax; doing so makes it easier
to copy debug output into unit tests, for example.

## User-facing: `fmt::Display`

The `fmt::Display` trait is intended for user-facing output. It should:

* Be implemented for scalars, strings, and other basic types.
* Be implemented for generic wrappers like `Option<T>` or smart pointers, where
  the output can be wholly delegated to a *single* `fmt::Display` implementation
  on the underlying type.
* *Not* be implemented for generic containers like `Vec<T>` or even `Result<T, E>`,
  where there is no useful, general way to tailor the output for user consumption.
* Be implemented for *specific* user-defined types as useful for an application,
  with application-defined user-facing output. In particular, applications will
  often make their types implement `fmt::Display` specifically for use in
  `format` interpolation.
* Never panic.
* Avoid quotes, escapes, and so on unless specifically desired for a user-facing purpose.
* Require use of an explicit adapter (like the `display` method in
  `Path`) when it potentially looses significant information.

A common pattern for `fmt::Display` is to provide simple "adapters", which are
types wrapping another type for the sole purpose of formatting in a certain
style or context. For example:

```rust
pub struct ForHtml<'a, T>(&'a T);
pub struct ForCli<'a, T>(&'a T);

impl MyInterestingType {
    fn for_html(&self) -> ForHtml<MyInterestingType> { ForHtml(self) }
    fn for_cli(&self) -> ForCli<MyInterestingType> { ForCli(self) }
}

impl<'a> fmt::Display for ForHtml<'a, MyInterestingType> { ... }
impl<'a> fmt::Display for ForCli<'a, MyInterestingType> { ... }
```

## Rationale for format specifiers

Given the above conventions, it should be clear that `fmt::Debug` is
much more commonly *implemented* on types than `fmt::Display`. Why,
then, use `{}` for `fmt::Display` and `{:?}` for `fmt::Debug`? Aren't
those the wrong defaults?

There are two main reasons for this choice:

* Debugging output usually makes very little use of interpolation. In general,
  one is typically using `#[derive(Show)]` or `format!("{:?}",
  something_to_debug)`, and the latter is better done via
  [more direct convenience](https://github.com/SimonSapin/rust-std-candidates#the-show-debugging-macro).

* When creating tailored string output via interpolation, the expected "default"
  formatting for things like strings is unquoted and unescapted. It would be
  surprising if the default specifiers below did not yield `"hello, world!" as the
  output string.

  ```rust
  format!("{}, {}!", "hello", "world")
  ```

In other words, although more types implement `fmt::Debug`, most
meaningful uses of interpolation (other than in such implementations)
will use `fmt::Display`, making `{}` the right choice.

## Use in errors

Right now, the (unstable) `Error` trait comes equipped with a `description`
method yielding an `Option<String>`. This RFC proposes to drop this method an
instead inherit from `fmt::Display`. It likewise proposes to make `unwrap` in
`Result` depend and use `fmt::Display` rather than `fmt::Debug`.

The reason in both cases is the same: although errors are often thought of in
terms of debugging, the messages they result in are often presented directly to
the user and should thus be tailored. Tying them to `fmt::Display` makes it
easier to remember and add such tailoring, and less likely to spew a lot of
unwanted internal representation.

# Alternatives

We've already explored an alternative where `Show` tries to play both of the
roles above, and found it to be problematic. There may, however, be alternative
conventions for a multi-trait world. The RFC author hopes this will emerge from
the discussion thread.

# Unresolved questions

(Previous questions here have been resolved in an RFC update).
