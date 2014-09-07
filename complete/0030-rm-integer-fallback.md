- Start Date: 2014-06-11
- RFC PR: [rust-lang/rfcs#115](https://github.com/rust-lang/rfcs/pull/115)
- Rust Issue: [rust-lang/rust#6023](https://github.com/rust-lang/rust/issues/6023)

# Summary

Currently we use inference to find the current type of
otherwise-unannotated integer literals, and when that fails the type
defaults to `int`. This is often felt to be potentially error-prone
behavior.

This proposal removes the integer inference fallback and strengthens
the types required for several language features that interact with
integer inference.

# Motivation

With the integer fallback, small changes to code can change the
inferred type in unexpected ways. It's not clear how big a problem
this is, but previous experiments[1] indicate that removing
the fallback has a relatively small impact on existing code,
so it's reasonable to back off of this feature in favor of more
strict typing.

See also https://github.com/mozilla/rust/issues/6023.

[1]: https://gist.github.com/nikomatsakis/11179747

# Detailed design

The primary change here is that, when integer type inference fails,
the compiler will emit an error instead of assigning the value the
type `int`.

This change alone will cause a fair bit of existing code to be
unable to type check because of lack of constraints. To add more
constraints and increase likelihood of unification, we 'tighten'
up what kinds of integers are required in some situations:

* Array repeat counts must be uint (`[expr, .. count]`)
* << and >> require uint when shifting integral types

Finally, inference for `as` will be modified to track the types
a value is being cast *to* for cases where the value being cast
is unconstrained, like `0 as u8`.

Treatment of enum discriminants will need to change:

```
enum Color { Red = 0, Green = 1, Blue = 2 }
```

Currently, an unsuffixed integer defaults to `int`. Instead, we will
only require enum descriminants primitive integers of unspecified
type; assigning an integer to an enum will behave as if casting from
from the type of the integer to an unsigned integer with the size of
the enum discriminant.

# Drawbacks

This will force users to type hint somewhat more often. In particular,
ranges of unsigned ints may need to be type-hinted:

```
for _ in range(0u, 10) { }
```

# Alternatives

Do none of this.

# Unresolved questions

* If we're putting new restrictions on shift operators, should we
  change the traits, or just make the primitives special?
