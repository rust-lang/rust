- Feature Name: `str_split_at`
- Start Date: 2015-05-17
- RFC PR: [rust-lang/rfcs#1123](https://github.com/rust-lang/rfcs/pull/1123)
- Rust Issue: [rust-lang/rust#25839](https://github.com/rust-lang/rust/pull/25839)

# Summary

Introduce the method `split_at(&self, mid: usize) -> (&str, &str)` on `str`,
to divide a slice into two, just like we can with `[T]`.

# Motivation

Adding `split_at` is a measure to provide a method from `[T]` in a version that
makes sense for `str`.

Once used to `[T]`, users might even expect that `split_at` is present on str.

It is a simple method with an obvious implementation, but it provides
convenience while working with string segmentation manually, which we already
have ample tools for (for example the method `find` that returns the first
matching byte offset).

Using `split_at` can lead to less repeated bounds checks, since it is easy to
use cumulatively, splitting off a piece at a time.

This feature is requested in [rust-lang/rust#18063][freq]

[freq]: https://github.com/rust-lang/rust/issues/18063

# Detailed design

Introduce the method `split_at(&self, mid: usize) -> (&str, &str)` on `str`, to
divide a slice into two.

`mid` will be a byte offset from the start of the string, and it must be on
a character boundary. Both `0` and `self.len()` are valid splitting points.

`split_at` will be an inherent method on `str` where possible, and will be
available from libcore and the layers above it.

The following is a working implementation, implemented as a trait just for
illustration and to be testable as a custom extension:

```rust
trait SplitAt {
    fn split_at(&self, mid: usize) -> (&Self, &Self);
}

impl SplitAt for str {
    /// Divide one string slice into two at an index.
    ///
    /// The index `mid` is a byte offset from the start of the string
    /// that must be on a character boundary.
    ///
    /// Return slices `&self[..mid]` and `&self[mid..]`.
    ///
    /// # Panics
    ///
    /// Panics if `mid` is beyond the last character of the string,
    /// or if it is not on a character boundary.
    ///
    /// # Examples
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let first_space = s.find(' ').unwrap_or(s.len());
    /// let (a, b) = s.split_at(first_space);
    ///
    /// assert_eq!(a, "Löwe");
    /// assert_eq!(b, " 老虎 Léopard");
    /// ```
    fn split_at(&self, mid: usize) -> (&str, &str) {
        (&self[..mid], &self[mid..])
    }
}
```

`split_at` will use a byte offset (a.k.a byte index) to be consistent with
slicing and the offset used by interrogator methods such as `find` or iterators
such as `char_indices`. Byte offsets are our standard lightweight position
indicators that we use to support efficient operations on string slices.

Implementing `split_at_mut` is not relevant for `str` at this time.

# Drawbacks

* `split_at` panics on 1) index out of bounds 2) index not on character
  boundary.
* Possible name confusion with other `str` methods like `.split()`
* According to our developing API evolution and semver guidelines this is a
  breaking change but a (very) minor change. Adding methods is something we
  expect to be able to. (See [RFC PR #1105][pr1105]).

[pr1105]: https://github.com/rust-lang/rfcs/pull/1105

# Alternatives

* Recommend other splitting methods, like the split iterators.
* Stick to writing `(&foo[..mid], &foo[mid..])`

# Unresolved questions

* *None*
