- Feature Name: dotdot_in_patterns
- Start Date: 2016-02-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/1492
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Permit the `..` pattern fragment in more contexts.

# Motivation
[motivation]: #motivation

The pattern fragment `..` can be used in some patterns to denote several elements in list contexts.
However, it doesn't always compiles when used in such contexts.
One can expect the ability to match tuple variants like `V(u8, u8, u8)` with patterns like
`V(x, ..)` or `V(.., z)`, but the compiler rejects such patterns currently despite accepting
very similar `V(..)`.

This RFC is intended to "complete" the feature and make it work in all possible list contexts,
making the language a bit more convenient and consistent.

# Detailed design
[design]: #detailed-design

Let's list all the patterns currently existing in the language, that contain lists of subpatterns:

```
// Struct patterns.
S { field1, field2, ..., fieldN }

// Tuple struct patterns.
S(field1, field2, ..., fieldN)

// Tuple patterns.
(field1, field2, ..., fieldN)

// Slice patterns.
[elem1, elem2, ..., elemN]
```
In all the patterns above, except for struct patterns, field/element positions are significant.

Now list all the contexts that currently permit the `..` pattern fragment:
```
// Struct patterns, the last position.
S { subpat1, subpat2, .. }

// Tuple struct patterns, the last and the only position, no extra subpatterns allowed.
S(..)

// Slice patterns, the last position.
[subpat1, subpat2, ..]
// Slice patterns, the first position.
[.., subpatN-1, subpatN]
// Slice patterns, any other position.
[subpat1, .., subpatN]
// Slice patterns, any of the above with a subslice binding.
// (The binding is not actually a binding, but one more pattern bound to the sublist, but this is
// not important for our discussion.)
[subpat1, binding.., subpatN]
```
Something is obviously missing, let's fill in the missing parts.

```
// Struct patterns, the last position.
S { subpat1, subpat2, .. }
// **NOT PROPOSED**: Struct patterns, any position.
// Since named struct fields are not positional, there's essentially no sense in placing the `..`
// anywhere except for one conventionally chosen position (the last one) or in sublist bindings,
// so we don't propose extensions to struct patterns.
S { subpat1, .., subpatN }
// **NOT PROPOSED**: Struct patterns with bindings
S { subpat1, binding.., subpatN }

// Tuple struct patterns, the last and the only position, no extra subpatterns allowed.
S(..)
// **NEW**: Tuple struct patterns, any position.
S(subpat1, subpat2, ..)
S(.., subpatN-1, subpatN)
S(subpat1, .., subpatN)
// **NOT PROPOSED**: Struct patterns with bindings
S(subpat1, binding.., subpatN)

// **NEW**: Tuple patterns, any position.
(subpat1, subpat2, ..)
(.., subpatN-1, subpatN)
(subpat1, .., subpatN)
// **NOT PROPOSED**: Tuple patterns with bindings
(subpat1, binding.., subpatN)
```

Slice patterns are not covered in this RFC, but here is the syntax for reference:

```
// Slice patterns, the last position.
[subpat1, subpat2, ..]
// Slice patterns, the first position.
[.., subpatN-1, subpatN]
// Slice patterns, any other position.
[subpat1, .., subpatN]
// Slice patterns, any of the above with a subslice binding.
// By ref bindings are allowed, slices and subslices always have compatible layouts.
[subpat1, binding.., subpatN]
```

Trailing comma is not allowed after `..` in the last position by analogy with existing slice and
struct patterns.

This RFC is not critically important and can be rolled out in parts, for example, bare `..` first,
`..` with a sublist binding eventually.

# Drawbacks
[drawbacks]: #drawbacks

None.

# Alternatives
[alternatives]: #alternatives

Do not permit sublist bindings in tuples and tuple structs at all.

# Unresolved questions
[unresolved]: #unresolved-questions

Sublist binding syntax conflicts with possible exclusive range patterns
`begin .. end`/`begin..`/`..end`. This problem already exists for slice patterns and has to be
solved independently from extensions to `..`.
This RFC simply selects the same syntax that slice patterns already have.
