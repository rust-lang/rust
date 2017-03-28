- Feature Name: compile\_error\_macro
- Start Date: 2016-08-01
- RFC PR: [rust-lang/rfcs#1695](https://github.com/rust-lang/rfcs/pull/1695)
- Rust Issue: [rust-lang/rust#40872](https://github.com/rust-lang/rust/issues/40872)

# Summary
[summary]: #summary

This RFC proposes adding a new macro to `libcore`, `compile_error!` which will
unconditionally cause compilation to fail with the given error message when
encountered.

# Motivation
[motivation]: #motivation

Crates which work with macros or annotations such as `cfg` have no tools to
communicate error cases in a meaningful way on stable. For example, given the
following macro:

```rust
macro_rules! give_me_foo_or_bar {
    (foo) => {};
    (bar) => {};
}
```

when invoked with `baz`, the error message will be `error: no rules expected the
token baz`. In a real world scenario, this error may actually occur deep in a
stack of macro calls, with an even more confusing error message. With this RFC,
the macro author could provide the following:

```rust
macro_rules! give_me_foo_or_bar {
    (foo) => {};
    (bar) => {};
    ($x:ident) => {
        compile_error!("This macro only accepts `foo` or `bar`");
    }
}
```

When combined with attributes, this also provides a way for authors to validate
combinations of features.

```rust
#[cfg(not(any(feature = "postgresql", feature = "sqlite")))]
compile_error!("At least one backend must be used with this crate. \
    Please specify `features = ["postgresql"]` or `features = ["sqlite"]`")
```

# Detailed design
[design]: #detailed-design

The span given for the failure should be the invocation of the `compile_error!`
macro. The macro must take exactly one argument, which is a string literal. The
macro will then call `span_err` with the provided message on the expansion
context, and will not expand to any further code.

# Drawbacks
[drawbacks]: #drawbacks

None

# Alternatives
[alternatives]: #alternatives

Wait for the stabilization of procedural macros, at which point a crate could
provide this functionality.

# Unresolved questions
[unresolved]: #unresolved-questions

None
