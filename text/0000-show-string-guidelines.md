- Start Date: 2015-01-08
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

A [recent RFC](https://github.com/rust-lang/rfcs/pull/504) split what was
previously `fmt::Show` into two traits, `fmt::Show` and `fmt::String`, with
format specifiers `{:?}` and `{}` respectively.

That RFC did not, however, establish complete conventions for when to implement
which of the traits, nor what is expected from the output.  That's what this RFC
seeks to do.

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

## Debugging: `fmt::Show`

The `fmt::Show` trait is intended for debugging. It should:

* Be implemented on every type, usually via `#[derive(Show)]`.
* Never panic.
* Escape away control characters.
* Introduce quotes and other delimiters as necessary to give a clear
  representation of the data involved.
* Focus on the *runtime* aspects of a type; repeating information such as
  suffixes for integer literals is not generally useful since that data is
  readily available from the type definition.

It is *not* a goal for the debugging output to be valid Rust source.

## User-facing: `fmt::String`

The `fmt::String` trait is intended for user-facing output. It should:

* Be implemented for scalars, strings, and other basic types.
* Be implemented for generic wrappers like `Option<T>` or smart pointers, where
  the output can be wholly delegated to a *single* `fmt::String` implementation
  on the underlying type.
* *Not* be implemented for generic containers like `Vec<T>` or even `Result<T, E>`,
  where there is no useful, general way to tailor the output for user consumption.
* Be implemented for *specific* user-defined types as useful for an application,
  with application-defined user-facing output. In particular, applications will
  often make their types implement `fmt::String` specifically for use in
  `format` interpolation.
* Never panic.
* Avoid quotes, escapes, and so on unless specifically desired for a user-facing purpose.

A common pattern for `fmt::String` is to provide simple "adapters", which are
types wrapping another type for the sole purpose of formatting in a certain
style or context. For example:

```rust
pub struct ForHtml<'a, T>(&'a T);
pub struct ForCli<'a, T>(&'a T);

impl MyInterestingType {
    fn for_html(&self) -> ForHtml<MyInterestingType> { ForHtml(self) }
    fn for_cli(&self) -> ForCli<MyInterestingType> { ForCli(self) }
}

impl<'a> fmt::String for ForHtml<'a, MyInterestingType> { ... }
impl<'a> fmt::String for ForCli<'a, MyInterestingType> { ... }
```

## Rationale for format specifiers

Given the above conventions, it should be clear that `fmt::Show` is much more
common than `fmt::String`. Why, then, use `{}` for `fmt::String` and `{:?}` for
`fmt::Show`? Aren't those the wrong defaults?

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

# Alternatives

We've already explored an alternative where `Show` tries to play both of the
roles above, and found it to be problematic. There may, however, be alternative
conventions for a multi-trait world. The RFC author hopes this will emerge from
the discussion thread.

# Unresolved questions

Should we set stricter rules about the exact formatting of debugging output?
