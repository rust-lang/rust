- Feature Name: intrinsic-semantics
- Start Date: 2015-09-29
- RFC PR: https://github.com/rust-lang/rfcs/pull/1300
- Rust Issue: N/A

# Summary

Define the general semantics of intrinsic functions. This does not define the semantics of the
individual intrinsics, instead defines the semantics around intrinsic functions in general.

# Motivation

Intrinsics are currently poorly-specified in terms of how they function. This means they are a
cause of ICEs and general confusion. The poor specification of them also means discussion affecting
intrinsics gets mired in opinions about what intrinsics should be like and how they should act or
be implemented.

# Detailed design

Intrinsics are currently implemented by generating the code for the intrinsic at the call
site. This allows for intrinsics to be implemented much more efficiently in many cases. For
example, `transmute` is able to evaluate the input expression directly into the storage for the
result, removing a potential copy. This is the main idea of intrinsics, a way to generate code that
is otherwise inexpressible in Rust.

Keeping this in-place behaviour is desirable, so this RFC proposes that intrinsics should only be
usable as functions when called. This is not a change from the current behaviour, as you already
cannot use intrinsics as function pointers. Using an intrinsic in any way other than directly
calling should be considered an error.

Intrinsics should continue to be defined and declared the same way. The `rust-intrinsic` and
`platform-intrinsic` ABIs indicate that the function is an intrinsic function.

# Drawbacks

* Fewer bikesheds to paint.
* Doesn't allow intrinsics to be used as regular functions. (Note that this is not something we
  have evidence to suggest is a desired property, as it is currently the case anyway)

# Alternatives

* Allow coercion to regular functions and generate wrappers. This is similar to how we handle named
  tuple constructors. Doing this undermines the idea of intrinsics as a way of getting the compiler
  to generate specific code at the call-site however.
* Do nothing.

# Unresolved questions

None.
