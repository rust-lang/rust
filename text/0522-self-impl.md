- Start Date: 2014-12-13
- RFC PR: [522](https://github.com/rust-lang/rfcs/pull/522)
- Rust Issue: [20000](https://github.com/rust-lang/rust/issues/20000)

# Summary

Allow `Self` type to be used in impls.

# Motivation

Allows macros which operate on methods to do more, more easily without having to
rebuild the concrete self type. Macros could use the literal self type like
programmers do, but that requires extra machinery in the macro expansion code
and extra work by the macro author.

Allows easier copy and pasting of method signatures from trait declarations to
implementations.

Is more succinct where the self type is complex.

## Motivation for doing this now

I'm hitting the macro problem in a side project. I wrote and hope to land the
compiler code to make it work, but it is ugly and this is a much nicer solution.
It is also really easy to implement, and since it is just a desugaring, it
should not add any additional complexity to the compiler. Obviously, this should
not block 1.0.

# Detailed design

When used inside an impl, `Self` is desugared during syntactic expansion to the
concrete type being implemented. `Self` can be used anywhere the desugared type
could be used.

# Drawbacks

There are some advantages to being explicit about the self type where it is
possible - clarity and fewer type aliases.

# Alternatives

We could just force authors to use the concrete type as we do currently. This
would require macro expansion code to make available the concrete type (or the
whole impl AST) to macros working on methods. The macro author would then
extract/construct the self type and use it instead of `Self`.

# Unresolved questions

None.
