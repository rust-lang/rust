- Feature Name: `panic_safe_slicing`
- Start Date: 2015-10-16
- RFC PR: [rust-lang/rfcs#1679](https://github.com/rust-lang/rfcs/pull/1679)
- Rust Issue: [rust-lang/rfcs#35729](https://github.com/rust-lang/rust/issues/35729)

# Summary

Add "panic-safe" or "total" alternatives to the existing panicking indexing syntax.

# Motivation

`SliceExt::get` and `SliceExt::get_mut` can be thought as non-panicking versions of the simple
indexing syntax, `a[idx]`, and `SliceExt::get_unchecked` and `SliceExt::get_unchecked_mut` can
be thought of as unsafe versions with bounds checks elided. However, there is no such equivalent for
`a[start..end]`, `a[start..]`, or `a[..end]`. This RFC proposes such methods to fill the gap.

# Detailed design

The `get`, `get_mut`, `get_unchecked`, and `get_unchecked_mut` will be made generic over `usize`
as well as ranges of `usize` like slice's `Index` implementation currently is. This will allow e.g.
`a.get(start..end)` which will behave analagously to `a[start..end]`.

Because methods cannot be overloaded in an ad-hoc manner in the same way that traits may be
implemented, we introduce a `SliceIndex` trait which is implemented by types which can index into a
slice:
```rust
pub trait SliceIndex<T> {
    type Output: ?Sized;

    fn get(self, slice: &[T]) -> Option<&Self::Output>;
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output>;
    unsafe fn get_unchecked(self, slice: &[T]) -> &Self::Output;
    unsafe fn get_mut_unchecked(self, slice: &[T]) -> &mut Self::Output;
    fn index(self, slice: &[T]) -> &Self::Output;
    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output;
}

impl<T> SliceIndex<T> for usize {
    type Output = T;
    // ...
}

impl<T, R> SliceIndex<T> for R
    where R: RangeArgument<usize>
{
    type Output = [T];
    // ...
}
```

And then alter the `Index`, `IndexMut`, `get`, `get_mut`, `get_unchecked`, and `get_mut_unchecked`
implementations to be generic over `SliceIndex`:
```rust
impl<T> [T] {
    pub fn get<I>(&self, idx: I) -> Option<I::Output>
        where I: SliceIndex<T>
    {
        idx.get(self)
    }

    pub fn get_mut<I>(&mut self, idx: I) -> Option<I::Output>
        where I: SliceIndex<T>
    {
        idx.get_mut(self)
    }

    pub unsafe fn get_unchecked<I>(&self, idx: I) -> I::Output
        where I: SliceIndex<T>
    {
        idx.get_unchecked(self)
    }

    pub unsafe fn get_mut_unchecked<I>(&mut self, idx: I) -> I::Output
        where I: SliceIndex<T>
    {
        idx.get_mut_unchecked(self)
    }
}

impl<T, I> Index<I> for [T]
    where I: SliceIndex<T>
{
    type Output = I::Output;

    fn index(&self, idx: I) -> &I::Output {
        idx.index(self)
    }
}

impl<T, I> IndexMut<I> for [T]
    where I: SliceIndex<T>
{
    fn index_mut(&self, idx: I) -> &mut I::Output {
        idx.index_mut(self)
    }
}
```

# Drawbacks

- The `SliceIndex` trait is unfortunate - it's tuned for exactly the set of methods it's used by.
  It only exists because inherent methods cannot be overloaded the same way that trait
  implementations can be. It would most likely remain unstable indefinitely.
- Documentation may suffer. Rustdoc output currently explicitly shows each of the ways you can
  index a slice, while there will simply be a single generic implementation with this change. This
  may not be that bad, though. The doc block currently seems to provided the most valuable
  information to newcomers rather than the trait bound, and that will still be present with this
  change.

# Alternatives

- Stay as is.
- A previous version of this RFC introduced new `get_slice` etc methods rather than overloading
  `get` etc. This avoids the utility trait but is somewhat less ergonomic.
- Instead of one trait amalgamating all of the required methods, we could have one trait per
  method. This would open a more reasonable door to stabilizing those traits, but adds quite a lot
  more surface area. Replacing an unstable `SliceIndex` trait with a collection would be
  backwards compatible.

# Unresolved questions

None
