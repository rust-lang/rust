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

Add `get_range`, `get_range_mut`, `get_range_unchecked`, `get_range_unchecked_mut` to `SliceExt`.

`get_range` and `get_range_mut` may be implemented roughly as follows:

```rust
use std::ops::{RangeFrom, RangeTo, Range};
use std::slice::from_raw_parts;
use core::slice::SliceExt;

trait Rangeable<T: ?Sized> {
    fn start(&self, slice: &T) -> usize;
    fn end(&self, slice: &T) -> usize;
}

impl<T: SliceExt + ?Sized> Rangeable<T> for RangeFrom<usize> {
    fn start(&self, _: &T) -> usize { self.start }
    fn end(&self, slice: &T) -> usize { slice.len() }
}

impl<T: SliceExt + ?Sized> Rangeable<T> for RangeTo<usize> {
    fn start(&self, _: &T) -> usize { 0 }
    fn end(&self, _: &T) -> usize { self.end }
}

impl<T: SliceExt + ?Sized> Rangeable<T> for Range<usize> {
    fn start(&self, _: &T) -> usize { self.start }
    fn end(&self, _: &T) -> usize { self.end }
}

trait GetRangeExt: SliceExt {
    fn get_range<R: Rangeable<Self>>(&self, range: R) -> Option<&[Self::Item]>;
}

impl<T> GetRangeExt for [T] {
    fn get_range<R: Rangeable<Self>>(&self, range: R) -> Option<&[T]> {
        let start = range.start(self);
        let end = range.end(self);

        if start > end { return None; }
        if end > self.len() { return None; }

        unsafe { Some(from_raw_parts(self.as_ptr().offset(start as isize), end - start)) }
    }
}

fn main() {
    let a = [1, 2, 3, 4, 5];

    assert_eq!(a.get_range(1..), Some(&a[1..]));
    assert_eq!(a.get_range(..3), Some(&a[..3]));
    assert_eq!(a.get_range(2..5), Some(&a[2..5]));
    assert_eq!(a.get_range(..6), None);
    assert_eq!(a.get_range(4..2), None);
}
```

`get_range_unchecked` and `get_range_unchecked_mut` should be the unchecked versions of the methods
above.

# Drawbacks

- Are these methods worth adding to `std`? Are such use cases common to justify such extention?

# Alternatives

- Stay as is.
- Could there be any other (and better!) total functions that serve the similar purpose?

# Unresolved questions

- Naming, naming, naming: Is `get_range` the most suitable name? How about `get_slice`, or just
  `slice`? Or any others?
