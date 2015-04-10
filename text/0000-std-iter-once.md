- Start Date: 2015-1-30
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add a `once` function to `std::iter` to construct an iterator yielding a given value one time, and an `empty` function to construct an iterator yielding no values.

# Motivation

This is a common task when working with iterators. Currently, this can be done in many ways, most of which are unergonomic, do not work for all types (e.g. requiring Copy/Clone), or both. `once` and `empty` are simple to implement, simple to use, and simple to understand.

# Detailed design

`once` will return a new struct, `std::iter::Once<T>`, implementing Iterator<T>. Internally, `Once<T>` is simply a newtype wrapper around `std::option::IntoIter<T>`. The actual body of `once` is thus trivial:

```rust
pub struct Once<T>(std::option::IntoIter<T>);

pub fn once<T>(x: T) -> Once<T> {
	Once(
		Some(x).into_iter()
	)
}
```

`empty` is similar:

```rust
pub struct Empty<T>(std::option::IntoIter<T>);

pub fn empty<T>(x: T) -> Empty<T> {
	Empty(
		None.into_iter()
	)
}
```

These wrapper structs exist to allow future backwards-compatible changes, and hide the implementation. 

# Drawbacks

Although a tiny amount of code, it still does come with a testing, maintainance, etc. cost.

It's already possible to do this via `Some(x).into_iter()`, `std::iter::repeat(x).take(1)` (for `x: Clone`), `vec![x].into_iter()`, various contraptions involving `iterate`...

The existence of the `Once` struct is not technically necessary.

# Alternatives

There are already many, many alternatives to this- `Option::into_iter()`, `iterate`...

The `Once` struct could be not used, with `std::option::IntoIter` used instead.

# Unresolved questions

Naturally, `once` is fairly bikesheddable. `one_time`? `repeat_once`?

Are versions of `once` that return `&T`/`&mut T` desirable?
