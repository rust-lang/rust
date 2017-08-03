- Feature Name: allow_if_let_irrefutables
- Start Date: 2017-07-27
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Currently when using an if let statement and an irrefutable pattern (read always match) is used the compiler complains with an `E0162: irrefutable if-let pattern`.

# Motivation
[motivation]: #motivation

The use cases for this is in the creation of macros where patterns are allowed because to support the `_` patterns the code has to be rewritten to be both much larger and include a compiler allow.
The expected outcome is for irrefutable patterns to be compiled to a tautology and have the if block accept it as if it was `if true { }`.
To support this, currently you must do something roughly the following, which seems to counteract the benefit of having if-let and while-let in the spec.

```rust
if let $p = $val {
    $b
}
```
Cannot be used, so the original match must be. The `allow` is forced so that the warning does not appear to the user of it since `_` won't be matched if `$p` is irrefutable.

```rust
#[allow(unreachable_patterns)]
match $val {
    $p => { $b; },
    _ => ()
}
```


# Detailed design
[design]: #detailed-design

1. Change the compiler error `irrefutable if-let-pattern` and similar patterns to an `error-by-default` lint that can be disabled by an `#[allow]` statement

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This can be taught by changing the second version of (`The Book`)[https://doc.rust-lang.org/book/second-edition/ch18-02-refutability.html] to not explicitly say that it is not allowed.
Adding that it is a lint that can be disabled.

# Drawbacks
[drawbacks]: #drawbacks

It allows programmers to manually write the line `if let _ = expr { } else { }` which is generally obfuscating and not desirable. However, this will not be explicitly allowed with the `#[allow]`.

# Alternatives
[alternatives]: #alternatives

* The trivial alternative: Do nothing. As your motivation explains, this only matters for macros anyways plus there already is an acceptable workaround (match). Code that needs this frequently can just package this workaround in its own macro and be done.

# Unresolved questions
[unresolved]: #unresolved-questions
