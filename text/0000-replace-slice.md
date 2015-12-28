- Feature Name: replace-slice
- Start Date: 2015-12-28
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Add a `replace_slice` method to `Vec<T>` and `String` removes a range of elements,
and replaces it in place with a given sequence of values.
The new sequence does not necessarily have the same length as the range it replaces.

# Motivation
[motivation]: #motivation

An implementation of this operation is either slow or dangerous.

The slow way uses `Vec::drain`, and then `Vec::insert` repeatedly.
The latter part takes quadratic time:
potentially many elements after the replaced range are moved by one offset
potentially many times, once for each new element.

The dangerous way, detailed below, takes linear time
but involves unsafely moving generic values with `std::ptr::copy`.
This is non-trivial `unsafe` code, where a bug could lead to double-dropping elements
or exposing uninitialized elements.
(Or for `String`, breaking the UTF-8 invariant.)
It therefore benefits form having a shared, carefully-reviewed implementation
rather than leaving it to every potential user to do it themselves.

While it could be an external crate on crates.io,
this operation is general-purpose enough that I think it belongs in the standard library,
similar to `Vec::drain`.

# Detailed design
[design]: #detailed-design

An example implementation is below.

The proposal is to have inherent methods instead of extension traits.
(Traits are used to make this testable outside of `std`
and to make a point in Unresolved Questions below.)

```rust
#![feature(collections, collections_range, str_char)]

extern crate collections;

use collections::range::RangeArgument;
use std::ptr;

trait ReplaceVecSlice<T> {
    fn replace_slice<R, I>(&mut self, range: R, iterable: I)
    where R: RangeArgument<usize>, I: IntoIterator<Item=T>, I::IntoIter: ExactSizeIterator;
}

impl<T> ReplaceVecSlice<T> for Vec<T> {
    fn replace_slice<R, I>(&mut self, range: R, iterable: I)
    where R: RangeArgument<usize>, I: IntoIterator<Item=T>, I::IntoIter: ExactSizeIterator
    {
        let len = self.len();
        let range_start = *range.start().unwrap_or(&0);
        let range_end = *range.end().unwrap_or(&len);
        assert!(range_start <= range_end);
        assert!(range_end <= len);
        let mut iter = iterable.into_iter();
        // Overwrite range
        for i in range_start..range_end {
            if let Some(new_element) = iter.next() {
                unsafe {
                    *self.get_unchecked_mut(i) = new_element
                }
            } else {
                // Iterator shorter than range
                self.drain(i..range_end);
                return
            }
        }
        // Insert rest
        let iter_len = iter.len();
        let elements_after = len - range_end;
        let free_space_start = range_end;
        let free_space_end = free_space_start + iter_len;

        // FIXME: merge the reallocating case with the first ptr::copy below?
        self.reserve(iter_len);

        let p = self.as_mut_ptr();
        unsafe {
            // In case iter.next() panics, leak some elements rather than risk double-freeing them.
            self.set_len(free_space_start);
            // Shift everything over to make space (duplicating some elements).
            ptr::copy(p.offset(free_space_start as isize),
                      p.offset(free_space_end as isize),
                      elements_after);
            for i in free_space_start..free_space_end {
                if let Some(new_element) = iter.next() {
                    *self.get_unchecked_mut(i) = new_element
                } else {
                    // Iterator shorter than its ExactSizeIterator::len()
                    ptr::copy(p.offset(free_space_end as isize),
                              p.offset(i as isize),
                              elements_after);
                    self.set_len(i + elements_after);
                    return
                }
            }
            self.set_len(free_space_end + elements_after);
        }
        // Iterator longer than its ExactSizeIterator::len(), degenerate to quadratic time
        for (new_element, i) in iter.zip(free_space_end..) {
            self.insert(i, new_element);
        }
    }
}

trait ReplaceStringSlice {
    fn replace_slice<R>(&mut self, range: R, s: &str) where R: RangeArgument<usize>;
}

impl ReplaceStringSlice for String {
    fn replace_slice<R>(&mut self, range: R, s: &str) where R: RangeArgument<usize> {
        if let Some(&start) = range.start() {
            assert!(self.is_char_boundary(start));
        }
        if let Some(&end) = range.end() {
            assert!(self.is_char_boundary(end));
        }
        unsafe {
            self.as_mut_vec()
        }.replace_slice(range, s.bytes())
    }
}

#[test]
fn it_works() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.replace_slice(2..4, [10, 11, 12].iter().cloned());
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    v.replace_slice(1..3, Some(20));
    assert_eq!(v, &[1, 20, 11, 12, 5]);
    let mut s = "Hello, world!".to_owned();
    s.replace_slice(7.., "世界!");
    assert_eq!(s, "Hello, 世界!");
}

#[test]
#[should_panic]
fn char_boundary() {
    let mut s = "Hello, 世界!".to_owned();
    s.replace_slice(..8, "")
}
```

This implementation defends against `ExactSizeIterator::len()` being incorrect.
If `len()` is too high, it reserves more capacity than necessary
and does more copying than necessary,
but stays in linear time.
If `len()` is too low, the algorithm degenerates to quadratic time
using `Vec::insert` for each additional new element.

# Drawbacks
[drawbacks]: #drawbacks

Same as for any addition to `std`:
not every program needs it, and standard library growth has a maintainance cost.

# Alternatives
[alternatives]: #alternatives

* Status quo: leave it to every one who wants this to do it the slow way or the dangerous way.
* Publish a crate on crates.io.
  Individual crates tend to be not very discoverable,
  so not this situation would not be so different from the status quo.

# Unresolved questions
[unresolved]: #unresolved-questions

* Should the `ExactSizeIterator` bound be removed?
  The lower bound of `Iterator::size_hint` could be used instead of `ExactSizeIterator::len`,
  but the degenerate quadratic time case would become “normal”.
  With `ExactSizeIterator` it only happens when `ExactSizeIterator::len` is incorrect
  which means that someone is doing something wrong.

* Alternatively, should `replace_slice` panic when `ExactSizeIterator::len` is incorrect?

* It would be nice to be able to `Vec::replace_slice` with a slice
  without writing `.iter().cloned()` explicitly.
  This is possible with the same trick as for the `Extend` trait
  ([RFC 839](https://github.com/rust-lang/rfcs/blob/master/text/0839-embrace-extend-extinguish.md)):
  accept iterators of `&T` as well as iterators of `T`:

  ```rust
  impl<'a, T: 'a> ReplaceVecSlice<&'a T> for Vec<T> where T: Copy {
      fn replace_slice<R, I>(&mut self, range: R, iterable: I)
      where R: RangeArgument<usize>, I: IntoIterator<Item=&'a T>, I::IntoIter: ExactSizeIterator
      {
          self.replace_slice(range, iterable.into_iter().cloned())
      }
  }
  ```

  However, this trick can not be used with an inherent method instead of a trait.
  (By the way, what was the motivation for `Extend` being a trait rather than inherent methods,
  before RFC 839?)

* Naming.
  I accidentally typed `replace_range` instead of `replace_slice` several times
  while typing up this RFC.
