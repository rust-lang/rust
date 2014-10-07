- Start Date: (fill me in with today's date, 2014-09-30)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Removes the "virtual struct" (aka struct inheritance) feature, which
is currently feature gated.

# Motivation

Virtual structs were added experimentally prior to the RFC process as
a way of inheriting fields from one struct when defining a new struct.

The feature was introduced and remains behind a feature gate.

The motivations for removing this feature altogether are:

1. The feature is likely to be replaced by a more general mechanism,
   as part of the need to address hierarchies such as the DOM, ASTs,
   and so on. See
   [this post](http://discuss.rust-lang.org/t/summary-of-efficient-inheritance-rfcs/494/43)
   for some recent discussion.

2. The implementation is somewhat buggy and incomplete, and the
   feature is not well-documented.

3. Although it's behind a feature gate, keeping the feature around is
   still a maintenance burden.

# Detailed design

Remove the implementation and feature gate for virtual structs.

Retain the `virtual` keyword as reserved for possible future use.

# Drawbacks

The language will no longer offer any built-in mechanism for avoiding
repetition of struct fields. Macros offer a reasonable workaround
until a more general mechanism is added.

# Unresolved questions

None known.
