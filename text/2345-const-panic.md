- Feature Name: `const_panic`
- Start Date: 2018-02-22
- RFC PR: [rust-lang/rfcs#2345](https://github.com/rust-lang/rfcs/pull/2345)
- Rust Issue: [rust-lang/rust#51999](https://github.com/rust-lang/rust/issues/51999)

# Summary
[summary]: #summary

Allow the use of `panic!`, `assert!` and `assert_eq!` within constants and
report their evaluation as a compile-time error.

# Motivation
[motivation]: #motivation

It can often be desirable to terminate a constant evaluation due to invalid
arguments. Currently there's no way to do this other than to use `Result` to
produce an `Err` in case of errors. Unfortunately this will end up as a runtime
problem and not abort compilation, even though the problem has been detected at
compile-time. There are already ways to abort compilation, e.g. by invoking
`["some assert failed"][42]` within a constant, which will abort with a
compile-time error pointing at the span of the index operation. But this hack is
not very convenient to use and produces the wrong error message.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

You can now use `panic!` and `assert!` within `const fn`s. This means that when
the const fn is invoked at runtime, you will get a regular panic, but if it is
invoked at compile-time, the panic message will show up as an error message.

As an example, imagine a function that converts strings to their corresponding
booleans.

```rust
const fn parse_bool(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        other => panic!("`{}` is not a valid bool"),
    }
}
parse_bool("true");
parse_bool("false");
parse_bool("foo");
```

will produce an error with your custom error message:

```
error[E0080]: `foo` is not a valid bool
 --> src/main.rs: 5:25
  |
5 |        other => panic!("`{}` is not a valid bool"),
  |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
note: during the evaluation of
   |
10 | parse_bool("foo");
   | ^^^^^^^^^^^^^^^^^
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

MIR interpretation gets a special case for the panic machinery (which isn't
const fn). If the `panic` lang item is entered, instead of producing an error
about it not being const fn, we produce a specialized error with the panic's
message. This panic reporting machinery is already present in the mir
interpreter, but needs the lang item detection in order to work.

Note that this internal machinery is inherently unstable and thus never
invoked directly by users. Users will use the `panic!` macro as an entry point.
The internal details of the panic handling might change in the future, but always
in a way that will keep allowing MIR interpretation to evaluate it. All future
changes will have to address this directly and regression tests should ensure
that we never break the const evaluability.

# Drawbacks
[drawbacks]: #drawbacks

We have to implement some magic around processing `fmt::Arguments` objects and
producing the panic message from that.

# Rationale and alternatives
[alternatives]: #alternatives

* We could add a special constant error reporting mechanism. This has the
  disadvantage of widening the gap between const eval and runtime execution.
* We could make `String` and formatting const enough to allow the panic
  formatting machinery to be interpreted and made const fn
* Don't produce a good error message, just say "const eval encountered an error"
  and point the user to the panic location. This already works out of the box
  right now. We can improve the error message in the future with the `String` +
  formatting alternative. This is the most minimalistic alternative to this RFC

# Unresolved questions
[unresolved]: #unresolved-questions

* Should there be some additional message in the error about this being a panic
  turned error? Or do we just produce the exact message the panic would produce?

* This change becomes really useful if `Result::unwrap` and `Option::unwrap`
  become const fn, doing both in one go might be a good idea
