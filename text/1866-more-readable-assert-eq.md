- Feature Name: more-readable-assert-eq
- Start Date: 2017-01-23
- RFC PR: https://github.com/rust-lang/rfcs/pull/1866
- Rust Issue: https://github.com/rust-lang/rust/issues/41615


# Summary
[summary]: #summary

Improve the `assert_eq` failure message formatting to increase legibility.

[Previous RFC issue](https://github.com/rust-lang/rfcs/issues/1864).


# Motivation
[motivation]: #motivation

Currently when `assert_eq` fails the default panic text has all the
information on one long line, which is difficult to parse. This is more
difficult when working with larger data structures. I'd like to alter the
format of this text in order improve legibility, putting each piece of
information on a different line.


# Detailed design
[design]: #detailed-design

Here is an failing test with the current format:

```
---- log_packet::tests::syntax_error_test stdout ----
        thread 'log_packet::tests::syntax_error_test' panicked at 'assertion failed: `(left == right)` (left: `"Syntax Error: a.rb:1: syntax error, unexpected end-of-input\n\n"`, right: `"Syntax error: a.rb:1: syntax error, unexpected end-of-input\n\n"`)', src/log_packet.rs:102
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```

Here is a failing test with an alternate format:

```
---- log_packet::tests::syntax_error_test stdout ----
        thread 'log_packet::tests::syntax_error_test' panicked at 'assertion failed: `(left == right)`

left:  `"Syntax Error: a.rb:1: syntax error, unexpected end-of-input\n\n"`
right: `"Syntax error: a.rb:1: syntax error, unexpected end-of-input\n\n"`

', src/log_packet.rs:102
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```

In addition to putting each expression on a separate line I've also padding
the word "left" with an extra space. This makes the values line up and easier
to visually diff.

This could be further improved with coloured diff'ing or indication of
differences. i.e. If two strings are between a certain levenshtein distance
colour additional chars green and missing ones red.

Here is a screenshot of the output of the Elixir lang ExUnit test assertion
macro, which I think is extremely clear:

![2017-01-22-232834_932x347_scrot](https://cloud.githubusercontent.com/assets/6134406/22187245/a862ea0a-e0fa-11e6-8861-2a7c08df4332.png)

As the stdlib does not contain any terminal colour manipulation features at
the moment LLVM style arrows could also be used, as suggested by @p-kraszewski:

```
---- log_packet::tests::syntax_error_test stdout ----
        thread 'log_packet::tests::syntax_error_test' panicked at 'assertion failed: `(left == right)`

left:  `"Syntax Error: a.rb:1: syntax error, unexpected end-of-input\n\n"`
right: `"Syntax error: a.rb:1: syntax error, unexpected end-of-input\n\n"`
         ~~~~~~ ^ ~~~~
', src/log_packet.rs:102
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```


# Drawbacks
[drawbacks]: #drawbacks

This could be a breaking change if people are parsing this text. I feel the
format of this text shouldn't be relied upon, so this is probably OK.

Colour diffing will require quite a bit more work to support terminals on all
platforms.


# Unresolved questions
[unresolved]: #unresolved-questions
