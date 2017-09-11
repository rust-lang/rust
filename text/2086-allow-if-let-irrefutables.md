- Feature Name: allow_if_let_irrefutables
- Start Date: 2017-07-27
- RFC PR: https://github.com/rust-lang/rfcs/pull/2086
- Rust Issue: https://github.com/rust-lang/rust/issues/44495

# Summary
[summary]: #summary

Currently when using an if let statement and an irrefutable pattern (read always match) is used the compiler complains with an `E0162: irrefutable if-let pattern`.
The current state breaks macros who want to accept patterns generically and this RFC proposes changing this error to an error-by-default lint which is allowed to be disabled by such macros.

# Motivation
[motivation]: #motivation

The use cases for this is in the creation of macros where patterns are allowed because to support the `_` patterns the code has to be rewritten to be both much larger and include an \[#allow\] statement for a lint that does not seem to be related to the problem.
The expected outcome is for irrefutable patterns to be compiled to a tautology and have the if block accept it as if it was `if true { }`.
To support this, currently you must do something roughly the following, which seems to counteract the benefit of having if-let and while-let in the spec.

```rust
#[allow(unreachable_patterns)]
match $val {
    $p => { $b; },
    _ => ()
}
```
The following cannot be used, so the previous must be. An `#[allow(irrefutable_let_pattern)]` is used so that the error-by-default lint does not appear to the user.

```rust
if let $p = $val {
    $b
}
```

# Detailed design
[design]: #detailed-design

1. Change the compiler error `irrefutable if-let-pattern` and similar patterns to an `error-by-default` lint that can be disabled by an `#[allow]` statement
2. Proposed lint name: `irrefutable_let_pattern`

Code Example (explicit):
```rust
#[allow(irrefutable_let_pattern)]
if let _ = 'a' {
    println!("Hello World");
}
```

Code Example (implicit):
```rust
macro_rules! check_five {
    ($p:pat) => {{
        #[allow(irrefutable_let_pattern)]
        if let $p = 5 {
            println!("Pattern matches five");
        }
    }};
}
```

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This can be taught by changing the second version of [The Book](https://doc.rust-lang.org/book/second-edition/ch18-02-refutability.html) to not explicitly say that it is not allowed.
Adding that it is a lint that can be disabled.

# Drawbacks
[drawbacks]: #drawbacks

It allows programmers to manually write the line `if let _ = expr { } else { }` which is generally obfuscating and not desirable. However, this will only be allowed with an explicit `#[allow(irrefutable_let_pattern)]`.

# Alternatives
[alternatives]: #alternatives

* The trivial alternative: Do nothing. As your motivation explains, this only matters for macros anyways plus there already is an acceptable workaround (match). Code that needs this frequently can just package this workaround in its own macro and be done.

# Unresolved questions
[unresolved]: #unresolved-questions
