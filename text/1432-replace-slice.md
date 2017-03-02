- Feature Name: splice
- Start Date: 2015-12-28
- RFC PR: [rust-lang/rfcs#1432](https://github.com/rust-lang/rfcs/pull/1432)
- Rust Issue: [rust-lang/rust#32310](https://github.com/rust-lang/rust/issues/32310)

# Summary
[summary]: #summary

Add a `splice` method to `Vec<T>` and `String` removes a range of elements,
and replaces it in place with a given sequence of values.
The new sequence does not necessarily have the same length as the range it replaces.
In the `Vec` case, this method returns an iterator of the elements being moved out, like `drain`.


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

trait VecSplice<T> {
    fn splice<R, I>(&mut self, range: R, iterable: I) -> Splice<I>
    where R: RangeArgument<usize>, I: IntoIterator<Item=T>;
}

impl<T> VecSplice<T> for Vec<T> {
    fn splice<R, I>(&mut self, range: R, iterable: I) -> Splice<I>
    where R: RangeArgument<usize>, I: IntoIterator<Item=T>
    {
        unimplemented!() // FIXME: Fill in when exact semantics are decided.
    }
}

struct Splice<I: IntoIterator> {
    vec: &mut Vec<I::Item>,
    range: Range<usize>
    iter: I::IntoIter,
    // FIXME: Fill in when exact semantics are decided.
}

impl<I: IntoIterator> Iterator for Splice<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!() // FIXME: Fill in when exact semantics are decided.
    }
}

impl<I: IntoIterator> Drop for Splice<I> {
    fn drop(&mut self) {
        unimplemented!() // FIXME: Fill in when exact semantics are decided.
    }
}

trait StringSplice {
    fn splice<R>(&mut self, range: R, s: &str) where R: RangeArgument<usize>;
}

impl StringSplice for String {
    fn splice<R>(&mut self, range: R, s: &str) where R: RangeArgument<usize> {
        if let Some(&start) = range.start() {
            assert!(self.is_char_boundary(start));
        }
        if let Some(&end) = range.end() {
            assert!(self.is_char_boundary(end));
        }
        unsafe {
            self.as_mut_vec()
        }.splice(range, s.bytes())
    }
}

#[test]
fn it_works() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.splice(2..4, [10, 11, 12].iter().cloned());
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    v.splice(1..3, Some(20));
    assert_eq!(v, &[1, 20, 11, 12, 5]);
    let mut s = "Hello, world!".to_owned();
    s.splice(7.., "世界!");
    assert_eq!(s, "Hello, 世界!");
}

#[test]
#[should_panic]
fn char_boundary() {
    let mut s = "Hello, 世界!".to_owned();
    s.splice(..8, "")
}
```

The elements of the vector after the range first be moved by an offset of
the lower bound of `Iterator::size_hint` minus the length of the range.
Then, depending on the real length of the iterator:

* If it’s the same as the lower bound, we’re done.
* If it’s lower than the lower bound (which was then incorrect), the elements will be moved once more.
* If it’s higher, the extra iterator items well be collected into a temporary `Vec`
  in order to know exactly how many there are, and the elements after will be moved once more.

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

* Should the input iterator be consumed incrementally at each `Splice::next` call,
  or only in `Splice::drop`?

* It would be nice to be able to `Vec::splice` with a slice
  without writing `.iter().cloned()` explicitly.
  This is possible with the same trick as for the `Extend` trait
  ([RFC 839](https://github.com/rust-lang/rfcs/blob/master/text/0839-embrace-extend-extinguish.md)):
  accept iterators of `&T` as well as iterators of `T`:

  ```rust
  impl<'a, T: 'a> VecSplice<&'a T> for Vec<T> where T: Copy {
      fn splice<R, I>(&mut self, range: R, iterable: I)
      where R: RangeArgument<usize>, I: IntoIterator<Item=&'a T>
      {
          self.splice(range, iterable.into_iter().cloned())
      }
  }
  ```

  However, this trick can not be used with an inherent method instead of a trait.
  (By the way, what was the motivation for `Extend` being a trait rather than inherent methods,
  before RFC 839?)

* If coherence rules and backward-compatibility allow it,
  this functionality could be added to `Vec::insert` and `String::insert`
  by overloading them / making them more generic.
  This would probably require implementing `RangeArgument` for `usize`
  representing an empty range,
  though a range of length 1 would maybe make more sense for `Vec::drain`
  (another user of `RangeArgument`).
