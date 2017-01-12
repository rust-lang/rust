- Feature Name: non_static_type_id
- Start Date: 2017-01-08
- RFC PR: 
- Rust Issue: 

# Summary
[summary]: #summary

Remove the `'static` bound from the `type_id` intrinsic so users can experiment with reflection over non-static types.

# Motivation
[motivation]: #motivation

A common method for storing a map of arbitrary data is to use a `HashMap<TypeId, Box<Any>>` and store data of type `T` using `TypeId::of::<T>()`. The problem is that a `TypeId` can only be constructed for a static type. It's not possible to generalise the type map to a `HashMap<TypeId, Box<Any + 'a>>`. This is a reasonable constraint for the stable API, because lifetimes may need to play a part in equality checks by type id. 

One workaround is to define a trait with an associated type that's expected to be a `'static` version of the implementor:

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

The `TypeId` for some type `T` can then be taken from `<T as Keyed>::Key`. However this trait needs to be marked as unsafe because it could lead to undefined behaviour if `T::Key` is not unique. It puts the burden on the caller to implement correctly for all types they want to store in the type map.

This RFC proposes simply removing the `'static` bound from the `type_id` intrinsic, leaving the stable `TypeId` and `Any` traits unchanged. That way users who opt-in to unstable intrinsics can build the type equality guarantees they need without waiting for stable API support.

This is an important first step in expanding the tools available to users at runtime to reason about their data. With the ability to fetch a guaranteed unique type id for non-static types, users can build their own `TypeId` or `Any` traits.

# Detailed design
[design]: #detailed-design

Remove the `'static` bound from the `type_id` intrinsic in `libcore`.

Allowing type ids for non-static types exposes the fact that concrete lifetimes aren't taken into account. This means a type id for `SomeStruct<'a, 'b>` will be the same as `SomeStruct<'static, 'static>`, even though they may be different types.

We can work around this issue by documenting that the `type_id` intrinsic doesn't consider lifetimes. When requesting a type id for `SomeStruct<'a, 'b>`, a caller recieves an id for `SomeStruct<'_, '_>`.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This changes an unstable compiler intrinsic so we don't need to teach it.

# Drawbacks
[drawbacks]: #drawbacks

Removing the `'static` bound means callers may now depend on the fact that `type_id` doesn't consider concrete lifetimes, even though this probably isn't its intended final behaviour.

# Alternatives
[alternatives]: #alternatives

Create a new intrinsic called `runtime_type_id` that's specifically designed ignore concrete lifetimes, like `type_id` does now. Having a totally separate intrinsic means `type_id` could be changed in the future to account for lifetimes without impacting the usecases that specifically ignore them.

# Unresolved questions
[unresolved]: #unresolved-questions
