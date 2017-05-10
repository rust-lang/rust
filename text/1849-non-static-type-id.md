- Feature Name: non_static_type_id
- Start Date: 2017-01-08
- RFC PR: [rust-lang/rfcs#1849](https://github.com/rust-lang/rfcs/pull/1849)
- Rust Issue: [rust-lang/rust#41875](https://github.com/rust-lang/rust/issues/41875)

# Summary
[summary]: #summary

Remove the `'static` bound from the `type_id` intrinsic so users can experiment with usecases where lifetimes either soundly irrelevant to type checking or where lifetime correctness is enforced elsewhere in the program.

# Motivation
[motivation]: #motivation

Sometimes it's useful to encode a type so it can be checked at runtime. This can be done using the `type_id` intrinsic, that gives an id value that's guaranteed to be unique across the types available to the program. The drawback is that it's only valid for types that are `'static`, because concrete lifetimes aren't encoded in the id. For most cases this makes sense, otherwise the encoded type could be used to represent data in lifetimes it isn't valid for. There are cases though where lifetimes can be soundly checked outside the type id, so it's not possible to misrepresent the validy of the data. These cases can't make use of type ids right now, they need to rely on workarounds. One such workaround is to define a trait with an associated type that's expected to be a `'static` version of the implementor:

```rust
unsafe trait Keyed {
	type Key: 'static;
}

struct NonStaticStruct<'a> {
	a: &'a str
}
unsafe impl <'a> Keyed for NonStaticStruct<'a> {
	type Key = NonStaticStruct<'static>;
}
```

This requires additional boilerplate that may lead to undefined behaviour if implemented incorrectly or not kept up to date.

This RFC proposes simply removing the `'static` bound from the `type_id` intrinsic, leaving the stable `TypeId` and `Any` traits unchanged. That way users who opt-in to unstable intrinsics can build the type equality guarantees they need without waiting for stable API support.

This is an important first step in expanding the tools available to users at runtime to reason about their data. With the ability to fetch a guaranteed unique type id for non-static types, users can build their own `TypeId` or `Any` traits.

# Detailed design
[design]: #detailed-design

Remove the `'static` bound from the `type_id` intrinsic in `libcore`.

Allowing type ids for non-static types exposes the fact that concrete lifetimes aren't taken into account. This means a type id for `SomeStruct<'a, 'b>` will be the same as `SomeStruct<'b, 'a>`, even though they're different types.

Users need to be very careful using `type_id` directly, because it can easily lead to undefined behaviour if lifetimes aren't verified properly.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This changes an unstable compiler intrinsic so we don't need to teach it. The change does need to come with plenty of warning that it's unsound for type-checking and can't be used to produce something like a lifetime parameterised `Any` trait.

# Drawbacks
[drawbacks]: #drawbacks

Removing the `'static` bound means callers may now depend on the fact that `type_id` doesn't consider concrete lifetimes, even though this probably isn't its intended final behaviour.

# Alternatives
[alternatives]: #alternatives

- Create a new intrinsic called `runtime_type_id` that's specifically designed ignore concrete lifetimes, like `type_id` does now. Having a totally separate intrinsic means `type_id` could be changed in the future to account for lifetimes without impacting the usecases that specifically ignore them.
- Don't do this. Stick with existing workarounds for getting a `TypeId` for non-static types.

# Unresolved questions
[unresolved]: #unresolved-questions
