- Feature Name: const_locals
- Start Date: 2018-01-11
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow `let` bindings in the body of constants and const fns. Additionally enable
destructuring in `let` bindings and const fn arguments.

# Motivation
[motivation]: #motivation

It makes writing const fns much more like writing regular functions and is
not possible right now because the old constant evaluator was a constant folder
that could only process expressions. With the miri const evaluator this feature
exists but is still disallowed.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

`let` bindings in constants and const fn work just like `let` bindings
everywhere else. Historically these did not exist in constants and const fn
because it would have been very hard to support them in the old const evaluator.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Expressions like `a + b + c` are already transformed to

```rust
let tmp = a + b;
tmp + c
```

With this RFC we can create bindings ourselves instead of only allowing compiler
generated bindings.

# Drawbacks
[drawbacks]: #drawbacks

You can create mutable locals in constants and then actually modify them. This
has no real impact on the constness, as the mutation happens entirely at compile
time and results in an immutable value.

# Rationale and alternatives
[alternatives]: #alternatives

The backend already supports this 100%. This is essentially just disabling a
check

## Why is this design the best in the space of possible designs?

Being the only design makes it the best design by definition

## What is the impact of not doing this?

Not having locals and destructuring severely limits the functions that can be
turned into const fn and generally leads to unreadable const fns.

# Unresolved questions
[unresolved]: #unresolved-questions
