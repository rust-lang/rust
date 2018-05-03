- Feature Name: optional_error_description
- Start Date: 2017-11-29
- RFC PR: https://github.com/rust-lang/rfcs/pull/2230
- Rust Issue: (leave this empty)

# Default implementation of `Error::description()`
[summary]: #summary

Provide a default implementation of the `Error` trait's `description()` method to save users trouble of implementing this flawed method.

# Motivation
[motivation]: #motivation

The `description()` method is a waste of time for implementors and users of the `Error` trait. There's high overlap between description and `Display`, which creates redundant implementation work and confusion about relationship of these two ways of displaying the error.

The `description()` method can't easily return a formatted string with per-instance error description. That's a gotcha for novice users struggling with the borrow checker, and gotcha for users trying to display the error, because the `description()` is going to return a less informative message than the `Display` trait.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Let's steer users away from the `description()` method.

1. Change the `description()` documentation to suggest use of the `Display` trait instead.
2. Provide a default implementation of the `description()` so that the `Error` trait can be implemented without worrying about this method.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Users of the `Error` trait can then pretend this method does not exist.

# Drawbacks
[drawbacks]: #drawbacks

When users start omitting bespoke `description()` implementations, code that still uses this method will start getting default strings instead of human-written description. If this becomes a problem, the `description()` method can also be formally deprecated (with the `#[deprecated]` attribute). However, there's no urgency to remove existing implementations of `description()`, so this RFC does not propose formal deprecation at this time to avoid unnecessary warnings during the transition.

# Rationale and alternatives
[alternatives]: #alternatives

- Do nothing, and rely on 3rd party crates to improve usability of errors (e.g. various crates providing `Error`-implementing macros or the `Fail` trait).
- The default message returned by `description` could be different.
    - it could be a hardcoded generic string, e.g. `"error"`,
    - it could return `core::intrinsics::type_name::<Self>()`,
    - it could try to be nicer, e.g. use the type's doccomment as the description, or convert type name to a sentence (`FileNotFoundError` -> "error: file not found").

# Unresolved questions
[unresolved]: #unresolved-questions

None yet.
