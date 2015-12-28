- Feature Name: contains_method
- Start Date: 2015-12-28
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Implement a method, `contains()`, for `Range`, `RangeFrom`, and `RangeTo`, checking if a number is in the range.

Note that the alternatives are just as important as the main proposal.

# Motivation
[motivation]: #motivation

The motivation behind this is simple: To be able to write simpler and more expressive code. This RFC introduces a "syntactic sugar" without doing so.

# Detailed design
[design]: #detailed-design

Implement a method, `contains()`, for `Range`, `RangeFrom`, and `RangeTo`. This method will check if a number is bound by the range. It will yield a boolean based on the condition defined by the range.

The implementation is as follows (placed in libcore, and reexported by libstd):

```rust
use core::ops::{Range, RangeTo, RangeFrom};

impl<Idx> Range<Idx> where Idx: PartialOrd<Idx> {
    fn contains(&self, item: Idx) -> bool {
        self.start <= item && self.end > item
    }
}

impl<Idx> RangeTo<Idx> where Idx: PartialOrd<Idx> {
    fn contains(&self, item: Idx) -> bool {
        self.end > item
    }
}

impl<Idx> RangeFrom<Idx> where Idx: PartialOrd<Idx> {
    fn contains(&self, item: Idx) -> bool {
        self.start <= item
    }
}

```

# Drawbacks
[drawbacks]: #drawbacks

Lacks of generics (see Alternatives).

# Alternatives
[alternatives]: #alternatives

## Add a `Contains` trait

This trait provides the method `.contains()` and implements it for all the Range types.

## Add a `.contains(item: Self::Item)` iterator method

This method returns a boolean, telling if the iterator contains the item given as parameter. Using method specialization, this can achieve the same performance as the method suggested in this RFC.

This is more flexible, and provide better performance (due to specialization) than just passing a closure comparing the items to a `any()` method.

## Make `.any()` generic over a new trait

Call this trait, `ItemPattern<Item>`. This trait is implemented for `Item` and `FnMut(Item) -> bool`. This is, in a sense, similar to `std::str::pattern::Pattern`.

Then let `.any()` generic over this trait (`T: ItemPattern<Self::Item>`) to allow `any()` taking `Self::Item` searching through the iterator for this particular value.

This will not achieve the same performance as the other proposals.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
