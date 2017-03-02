- Start Date: 2015-01-12
- RFC PR #: https://github.com/rust-lang/rfcs/pull/574
- Rust Issue #: https://github.com/rust-lang/rust/issues/23055

# Summary

Replace `Vec::drain` by a method that accepts a range parameter. Add
`String::drain` with similar functionality.

# Motivation

Allowing a range parameter is strictly more powerful than the current version.
E.g., see the following implementations of some `Vec` methods via the hypothetical
`drain_range` method:

```rust
fn truncate(x: &mut Vec<u8>, len: usize) {
    if len <= x.len() {
        x.drain_range(len..);
    }
}

fn remove(x: &mut Vec<u8>, index: usize) -> u8 {
    x.drain_range(index).next().unwrap()
}

fn pop(x: &mut Vec<u8>) -> Option<u8> {
    match x.len() {
        0 => None,
        n => x.drain_range(n-1).next()
    }
}

fn drain(x: &mut Vec<u8>) -> DrainRange<u8> {
    x.drain_range(0..)
}

fn clear(x: &mut Vec<u8>) {
    x.drain_range(0..);
}
```

With optimization enabled, those methods will produce code that runs as fast
as the current versions. (They should not be implemented this way.)

In particular, this method allows the user to remove a slice from a vector in
`O(Vec::len)` instead of `O(Slice::len * Vec::len)`.

# Detailed design

Remove `Vec::drain` and add the following method:

```rust
/// Creates a draining iterator that clears the specified range in the Vec and
/// iterates over the removed items from start to end.
///
/// # Panics
///
/// Panics if the range is decreasing or if the upper bound is larger than the
/// length of the vector.
pub fn drain<T: Trait>(&mut self, range: T) -> /* ... */;
```

Where `Trait` is some trait that is implemented for at least `Range<usize>`,
`RangeTo<usize>`, `RangeFrom<usize>`, `FullRange`, and `usize`.

The precise nature of the return value is to be determined during implementation
and may or may not depend on `T`.

Add `String::drain`:

```rust
/// Creates a draining iterator that clears the specified range in the String
/// and iterates over the characters contained in the range.
///
/// # Panics
///
/// Panics if the range is decreasing, if the upper bound is larger than the
/// length of the String, or if the start and the end of the range don't lie on
/// character boundaries.
pub fn drain<T: Trait>(&mut self, range: T) -> /* ... */;
```

Where `Trait` and the return value are as above but need not be the same.

# Drawbacks

- The function signature differs from other collections.
- It's not clear from the signature that `..` can be used to get the old behavior.
- The trait documentation will link to the `std::ops` module. It's not immediately apparent how the types in there are related to the `N..M` syntax.
- Some of these problems can be mitigated by solid documentation of the function itself.
