- Feature Name: `contains_method_for_various_collections`
- Start Date: 2016-03-16
- RFC PR: [rust-lang/rfcs#1552](https://github.com/rust-lang/rfcs/pull/1552)
- Rust Issue: [rust-lang/rust#32630](https://github.com/rust-lang/rust/issues/32630)

# Summary
[summary]: #summary

Add a `contains` method to `VecDeque` and `LinkedList` that checks if the
collection contains a given item.

# Motivation
[motivation]: #motivation

A `contains` method exists for the slice type `[T]` and for `Vec` through
`Deref`, but there is no easy way to check if a `VecDeque` or `LinkedList`
contains a specific item. Currently, the shortest way to do it is something
like:

```rust
vec_deque.iter().any(|e| e == item)
```

While this is not insanely verbose, a `contains` method has the following
advantages:

- the name `contains` expresses the programmer's intent...
- ... and thus is more idiomatic
- it's as short as it can get
- programmers that are used to call `contains` on a `Vec` are confused by the
  non-existence of the method for `VecDeque` or `LinkedList`

# Detailed design
[design]: #detailed-design

Add the following method to `std::collections::VecDeque`:

```rust
impl<T> VecDeque<T> {
    /// Returns `true` if the `VecDeque` contains an element equal to the
    /// given value.
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq<T>
    {
        // implementation with a result equivalent to the result
        // of `self.iter().any(|e| e == x)`
    }
}
```

Add the following method to `std::collections::LinkedList`:

```rust
impl<T> LinkedList<T> {
    /// Returns `true` if the `LinkedList` contains an element equal to the
    /// given value.
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq<T>
    {
        // implementation with a result equivalent to the result
        // of `self.iter().any(|e| e == x)`
    }
}
```

The new methods should probably be marked as unstable initially and be
stabilized later.

# Drawbacks
[drawbacks]: #drawbacks

Obviously more methods increase the complexity of the standard library, but in
case of this RFC the increase is rather tiny.

While `VecDeque::contains` should be (nearly) as fast as `[T]::contains`,
`LinkedList::contains` will probably be much slower due to the cache
inefficient nature of a linked list. Offering a method that is short to
write and convenient to use could lead to excessive use of said method
without knowing about the problems mentioned above.

# Alternatives
[alternatives]: #alternatives

There are a few alternatives:

- add `VecDeque::contains` only and do not add `LinkedList::contains`
- do nothing, because -- technically -- the same functionality is offered
  through iterators
- also add `BinaryHeap::contains`, since it could be convenient for some use
  cases, too

# Unresolved questions
[unresolved]: #unresolved-questions

None so far.
