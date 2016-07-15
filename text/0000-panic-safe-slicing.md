- Feature Name: panic_safe_slicing
- Start Date: 2015-10-16
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add "panic-safe" or "total" alternatives to the existing panicking slicing syntax.

# Motivation

`SliceExt::get` and `SliceExt::get_mut` can be thought as non-panicking versions of the simple
slicing syntax, `a[idx]`. However, there is no such equivalent for `a[start..end]`, `a[start..]`,
or `a[..end]`. This RFC proposes such methods to fill the gap.

# Detailed design

Introduce a `SliceIndex` trait which is implemented by types which can index into a slice:
```rust
pub trait SliceIndex<T> {
    type Output: ?Sized;

    fn get(self, slice: &[T]) -> Option<&Self::Output>;
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output>;
    unsafe fn get_unchecked(self, slice: &[T]) -> &Self::Output;
    unsafe fn get_mut_unchecked(self, slice: &[T]) -> &mut Self::Output;
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

Alter the `Index`, `IndexMut`, `get`, `get_mut`, `get_unchecked`, and `get_mut_unchecked`
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
        self.get(idx).expect("out of bounds slice access")
    }
}

impl<T, I> IndexMut<I> for [T]
    where I: SliceIndex<T>
{
    fn index_mut(&self, idx: I) -> &mut I::Output {
        self.get_mut(idx).expect("out of bounds slice access")
    }
}
```

# Drawbacks

- The `SliceIndex` trait is unfortunate - it's tuned for exactly the set of methods it's used by.
  It only exists because inherent methods cannot be overloaded the same way that trait
  implementations can be. It would most likely remain unstable indefinitely.

# Alternatives

- Stay as is.
- A previous version of this RFC introduced new `get_slice` etc methods rather than overloading
  `get` etc. This avoids the utility trait but is somewhat less ergonomic.

# Unresolved questions

None
