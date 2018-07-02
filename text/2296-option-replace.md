- Feature Name: `option-replace`
- Start Date: 2017-01-16
- RFC PR: [rust-lang/rfcs#2296](https://github.com/rust-lang/rfcs/pull/2296)
- Rust Issue: [rust-lang/rust#51998](https://github.com/rust-lang/rust/issues/51998)

# Summary
[summary]: #summary

This RFC proposes the addition of `Option::replace` to complete the `Option::take` method, it replaces the actual value in the option by `Some` with the value given in parameter, returning the old value if present, without deinitializing either one.

# Motivation
[motivation]: #motivation

You can see the `Option` as a container and other containers already have this kind of method to change a value in-place like the [HashMap::replace](https://doc.rust-lang.org/std/collections/struct.HashSet.html#method.replace) method.

How do you replace a value inside an `Option`, you can use `mem::replace` but it can be really unconvenient to import the `mem` module just for that. Why not adding a useful method to do that ?

This is the symmetry of the already present `Option::take` method.

# Detailed design
[design]: #detailed-design

This method will be added to the `core::option::Option` type implementation:

```rust
use core::mem::replace;

impl<T> Option<T> {
    // ...

    pub fn replace(&mut self, value: T) -> Option<T> {
        mem::replace(self, Some(value))
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

It increases the size of the standard library by a tiny bit.

The add of this method could be a breaking change in the case of an already implemented method on the `Option` enum with the `replace` name. (i.e. a Trait defining the `replace` method that has been implemented on the `Option` type).

This method behavior could be misinterpreted: Updating the `Option` only if the variant is `Some`, doing nothing if its `None`. This other method could exist too and be named `map_in_place` or `modify`, no method having this kind of behavior already exist in the Rust std library.

# Alternatives
[alternatives]: #alternatives

- Don't use the `replace` name and use `give` instead in symmetry with the actual `take` method.
- Use directly `mem::replace`.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
