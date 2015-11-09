- Feature Name: `repr_align`
- Start Date: 2015-11-09
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Extend the existing `#[repr]` attribute on structs with an `align = "N"` option
to specify a custom alignment for `struct` types.

# Motivation
[motivation]: #motivation

The alignment of a type is normally not worried about as the compiler will "do
the right thing" of picking an appropriate alignment for general use cases.
There are situations, however, where a nonstandard alignment may be desired when
operating with foreign systems. For example these sorts of situations tend to
necessitate or be much easier with a custom alignment:

* Hardware can often have obscure requirements such as "this structure is
  aligned to 32 bytes" when it in fact is only composed of 4-byte values. While
  this can typically be manually calculated and managed, it's often also useful
  to express this as a property of a type to get the compiler to do a little
  extra work instead.
* C compilers like gcc and clang offer the ability to specify a custom alignment
  for structures, and Rust can much more easily interoperate with these types if
  Rust can also mirror the request for a custom alignment (e.g. passing a
  structure to C correctly is much easier).
* Custom alignment can often be used for various tricks here and there and is
  often convenient as "let's play around with an implementation" tool. For
  example this can be used to statically allocate page tables in a kernel
  or create an at-least cache-line-sized structure easily for concurrent
  programming.

Currently these sort of situations are possible in Rust but aren't necessarily
the most ergonomic as programmers must manually manage alignment. The purpose of
this RFC is to provide a lightweight annotation to alter the compiler-inferred
alignment of a structure to enable these situations much more easily.

# Detailed design
[design]: #detailed-design

The `#[repr]` attribute on `struct`s will be extended to include a form such as:

```rust
#[repr(align = "16")]
struct MoreAligned(i32);
```

This structure will still have an alignment of 16 (as returned by
`mem::align_of`), and in this case the size will also be 16.

Syntactically, the `repr` meta list will be extended to accept a meta item
name/value pair with the name "align" and the value as a string which can be
parsed as a `u64`. The restrictions on where this attribute can be placed along
with the accepted values are:

* Custom alignment can only be specified on `struct` declarations for now.
  Specifying a different alignment on perhaps `enum` or `type` definitions
  should be a backwards-compatible extension.
* Alignment values must be a power of two.

A custom alignment cannot *decrease* the alignment of a structure unless it is
also declared with `#[repr(packed)]` (to mirror what C does in this regard), but
it can increase the alignment (and hence size) of a structure (as shown
above).

Semantically, it will be guaranteed (modulo `unsafe` code) that custom alignment
will always be respected. If a pointer to a non-aligned structure exists and is
used then it is considered unsafe behavior. Local variables, objects in arrays,
statics, etc, will all respect the custom alignment specified for a type.

# Drawbacks
[drawbacks]: #drawbacks

Specifying a custom alignment isn't always necessarily easy to do so via a
literal integer value. It may require usage of `#[cfg_attr]` in some situations
and may otherwise be much more convenient to name a different type instead.
Working with a raw integer, however, should provide the building block for
building up other abstractions and should be maximally flexible. It also
provides a relatively straightforward implementation and understanding of the
attribute at hand.

This also currently does not allow for specifying the custom alignment of a
struct field (as C compilers also allow doing) without the usage of a newtype
structure. Currently `#[repr]` is not recognized here, but it would be a
backwards compatible extension to start reading it on struct fields.

# Alternatives
[alternatives]: #alternatives

Instead of using the `#[repr]` attribute as the "house" for the custom
alignment, there could instead be a new `#[align = "..."]` attribute. This is
perhaps more extensible to alignment in other locations such as a local variable
(with attributes on expressions), a struct field (where `#[repr]` is more of an
"outer attribute"), or enum variants perhaps.

# Unresolved questions
[unresolved]: #unresolved-questions

* It is likely best to simply match the semantics of C/C++ in the regard of
  custom alignment, but is it ensured that this RFC is the same as the behavior
  of standard C compilers?
