// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::Option::{self, Some};
use marker::Sized;

use super::Iterator;

/// Conversion from an `Iterator`.
///
/// By implementing `FromIterator` for a type, you define how it will be
/// created from an iterator. This is common for types which describe a
/// collection of some kind.
///
/// `FromIterator`'s [`from_iter()`] is rarely called explicitly, and is instead
/// used through [`Iterator`]'s [`collect()`] method. See [`collect()`]'s
/// documentation for more examples.
///
/// [`from_iter()`]: #tymethod.from_iter
/// [`Iterator`]: trait.Iterator.html
/// [`collect()`]: trait.Iterator.html#method.collect
///
/// See also: [`IntoIterator`].
///
/// [`IntoIterator`]: trait.IntoIterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter::FromIterator;
///
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v = Vec::from_iter(five_fives);
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Using [`collect()`] to implicitly use `FromIterator`:
///
/// ```
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v: Vec<i32> = five_fives.collect();
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Implementing `FromIterator` for your type:
///
/// ```
/// use std::iter::FromIterator;
///
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement FromIterator
/// impl FromIterator<i32> for MyCollection {
///     fn from_iter<I: IntoIterator<Item=i32>>(iter: I) -> Self {
///         let mut c = MyCollection::new();
///
///         for i in iter {
///             c.add(i);
///         }
///
///         c
///     }
/// }
///
/// // Now we can make a new iterator...
/// let iter = (0..5).into_iter();
///
/// // ... and make a MyCollection out of it
/// let c = MyCollection::from_iter(iter);
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
///
/// // collect works too!
///
/// let iter = (0..5).into_iter();
/// let c: MyCollection = iter.collect();
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented="a collection of type `{Self}` cannot be \
                          built from an iterator over elements of type `{A}`"]
pub trait FromIterator<A>: Sized {
    /// Creates a value from an iterator.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: trait.FromIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::iter::FromIterator;
    ///
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_iter<T: IntoIterator<Item=A>>(iter: T) -> Self;
}

/// Conversion into an `Iterator`.
///
/// By implementing `IntoIterator` for a type, you define how it will be
/// converted to an iterator. This is common for types which describe a
/// collection of some kind.
///
/// One benefit of implementing `IntoIterator` is that your type will [work
/// with Rust's `for` loop syntax](index.html#for-loops-and-intoiterator).
///
/// See also: [`FromIterator`].
///
/// [`FromIterator`]: trait.FromIterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let v = vec![1, 2, 3];
///
/// let mut iter = v.into_iter();
///
/// let n = iter.next();
/// assert_eq!(Some(1), n);
///
/// let n = iter.next();
/// assert_eq!(Some(2), n);
///
/// let n = iter.next();
/// assert_eq!(Some(3), n);
///
/// let n = iter.next();
/// assert_eq!(None, n);
/// ```
///
/// Implementing `IntoIterator` for your type:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement IntoIterator
/// impl IntoIterator for MyCollection {
///     type Item = i32;
///     type IntoIter = ::std::vec::IntoIter<i32>;
///
///     fn into_iter(self) -> Self::IntoIter {
///         self.0.into_iter()
///     }
/// }
///
/// // Now we can make a new collection...
/// let mut c = MyCollection::new();
///
/// // ... add some stuff to it ...
/// c.add(0);
/// c.add(1);
/// c.add(2);
///
/// // ... and then turn it into an Iterator:
/// for (i, n) in c.into_iter().enumerate() {
///     assert_eq!(i as i32, n);
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IntoIterator {
    /// The type of the elements being iterated over.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Which kind of iterator are we turning this into?
    #[stable(feature = "rust1", since = "1.0.0")]
    type IntoIter: Iterator<Item=Self::Item>;

    /// Creates an iterator from a value.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: trait.IntoIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v = vec![1, 2, 3];
    ///
    /// let mut iter = v.into_iter();
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(1), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(2), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(3), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(None, n);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_iter(self) -> Self::IntoIter;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> IntoIterator for I {
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> I {
        self
    }
}

/// Extend a collection with the contents of an iterator.
///
/// Iterators produce a series of values, and collections can also be thought
/// of as a series of values. The `Extend` trait bridges this gap, allowing you
/// to extend a collection by including the contents of that iterator.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // You can extend a String with some chars:
/// let mut message = String::from("The first three letters are: ");
///
/// message.extend(&['a', 'b', 'c']);
///
/// assert_eq!("abc", &message[29..32]);
/// ```
///
/// Implementing `Extend`:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // since MyCollection has a list of i32s, we implement Extend for i32
/// impl Extend<i32> for MyCollection {
///
///     // This is a bit simpler with the concrete type signature: we can call
///     // extend on anything which can be turned into an Iterator which gives
///     // us i32s. Because we need i32s to put into MyCollection.
///     fn extend<T: IntoIterator<Item=i32>>(&mut self, iter: T) {
///
///         // The implementation is very straightforward: loop through the
///         // iterator, and add() each element to ourselves.
///         for elem in iter {
///             self.add(elem);
///         }
///     }
/// }
///
/// let mut c = MyCollection::new();
///
/// c.add(5);
/// c.add(6);
/// c.add(7);
///
/// // let's extend our collection with three more numbers
/// c.extend(vec![1, 2, 3]);
///
/// // we've added these elements onto the end
/// assert_eq!("MyCollection([5, 6, 7, 1, 2, 3])", format!("{:?}", c));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Extend<A> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// As this is the only method for this trait, the [trait-level] docs
    /// contain more details.
    ///
    /// [trait-level]: trait.Extend.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // You can extend a String with some chars:
    /// let mut message = String::from("abc");
    ///
    /// message.extend(['d', 'e', 'f'].iter());
    ///
    /// assert_eq!("abcdef", &message);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn extend<T: IntoIterator<Item=A>>(&mut self, iter: T);
}

/// An iterator able to yield elements from both ends.
///
/// Something that implements `DoubleEndedIterator` has one extra capability
/// over something that implements [`Iterator`]: the ability to also take
/// `Item`s from the back, as well as the front.
///
/// It is important to note that both back and forth work on the same range,
/// and do not cross: iteration is over when they meet in the middle.
///
/// In a similar fashion to the [`Iterator`] protocol, once a
/// `DoubleEndedIterator` returns `None` from a `next_back()`, calling it again
/// may or may not ever return `Some` again. `next()` and `next_back()` are
/// interchangable for this purpose.
///
/// [`Iterator`]: trait.Iterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let numbers = vec![1, 2, 3, 4, 5, 6];
///
/// let mut iter = numbers.iter();
///
/// assert_eq!(Some(&1), iter.next());
/// assert_eq!(Some(&6), iter.next_back());
/// assert_eq!(Some(&5), iter.next_back());
/// assert_eq!(Some(&2), iter.next());
/// assert_eq!(Some(&3), iter.next());
/// assert_eq!(Some(&4), iter.next());
/// assert_eq!(None, iter.next());
/// assert_eq!(None, iter.next_back());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DoubleEndedIterator: Iterator {
    /// An iterator able to yield elements from both ends.
    ///
    /// As this is the only method for this trait, the [trait-level] docs
    /// contain more details.
    ///
    /// [trait-level]: trait.DoubleEndedIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let numbers = vec![1, 2, 3, 4, 5, 6];
    ///
    /// let mut iter = numbers.iter();
    ///
    /// assert_eq!(Some(&1), iter.next());
    /// assert_eq!(Some(&6), iter.next_back());
    /// assert_eq!(Some(&5), iter.next_back());
    /// assert_eq!(Some(&2), iter.next());
    /// assert_eq!(Some(&3), iter.next());
    /// assert_eq!(Some(&4), iter.next());
    /// assert_eq!(None, iter.next());
    /// assert_eq!(None, iter.next_back());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next_back(&mut self) -> Option<Self::Item>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &'a mut I {
    fn next_back(&mut self) -> Option<I::Item> { (**self).next_back() }
}

/// An iterator that knows its exact length.
///
/// Many [`Iterator`]s don't know how many times they will iterate, but some do.
/// If an iterator knows how many times it can iterate, providing access to
/// that information can be useful. For example, if you want to iterate
/// backwards, a good start is to know where the end is.
///
/// When implementing an `ExactSizeIterator`, You must also implement
/// [`Iterator`]. When doing so, the implementation of [`size_hint()`] *must*
/// return the exact size of the iterator.
///
/// [`Iterator`]: trait.Iterator.html
/// [`size_hint()`]: trait.Iterator.html#method.size_hint
///
/// The [`len()`] method has a default implementation, so you usually shouldn't
/// implement it. However, you may be able to provide a more performant
/// implementation than the default, so overriding it in this case makes sense.
///
/// [`len()`]: #method.len
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // a finite range knows exactly how many times it will iterate
/// let five = 0..5;
///
/// assert_eq!(5, five.len());
/// ```
///
/// In the [module level docs][moddocs], we implemented an [`Iterator`],
/// `Counter`. Let's implement `ExactSizeIterator` for it as well:
///
/// [moddocs]: index.html
///
/// ```
/// # struct Counter {
/// #     count: usize,
/// # }
/// # impl Counter {
/// #     fn new() -> Counter {
/// #         Counter { count: 0 }
/// #     }
/// # }
/// # impl Iterator for Counter {
/// #     type Item = usize;
/// #     fn next(&mut self) -> Option<usize> {
/// #         self.count += 1;
/// #         if self.count < 6 {
/// #             Some(self.count)
/// #         } else {
/// #             None
/// #         }
/// #     }
/// # }
/// impl ExactSizeIterator for Counter {
///     // We already have the number of iterations, so we can use it directly.
///     fn len(&self) -> usize {
///         self.count
///     }
/// }
///
/// // And now we can use it!
///
/// let counter = Counter::new();
///
/// assert_eq!(0, counter.len());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExactSizeIterator: Iterator {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Returns the exact number of times the iterator will iterate.
    ///
    /// This method has a default implementation, so you usually should not
    /// implement it directly. However, if you can provide a more efficient
    /// implementation, you can do so. See the [trait-level] docs for an
    /// example.
    ///
    /// This function has the same safety guarantees as the [`size_hint()`]
    /// function.
    ///
    /// [trait-level]: trait.ExactSizeIterator.html
    /// [`size_hint()`]: trait.Iterator.html#method.size_hint
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // a finite range knows exactly how many times it will iterate
    /// let five = 0..5;
    ///
    /// assert_eq!(5, five.len());
    /// ```
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: ExactSizeIterator + ?Sized> ExactSizeIterator for &'a mut I {}

/// Trait to represent types that can be created by summing up an iterator.
///
/// This trait is used to implement the `sum` method on iterators. Types which
/// implement the trait can be generated by the `sum` method. Like
/// `FromIterator` this trait should rarely be called directly and instead
/// interacted with through `Iterator::sum`.
#[unstable(feature = "iter_arith_traits", issue = "34529")]
pub trait Sum<A = Self>: Sized {
    /// Method which takes an iterator and generates `Self` from the elements by
    /// "summing up" the items.
    fn sum<I: Iterator<Item=A>>(iter: I) -> Self;
}

/// Trait to represent types that can be created by multiplying elements of an
/// iterator.
///
/// This trait is used to implement the `product` method on iterators. Types
/// which implement the trait can be generated by the `product` method. Like
/// `FromIterator` this trait should rarely be called directly and instead
/// interacted with through `Iterator::product`.
#[unstable(feature = "iter_arith_traits", issue = "34529")]
pub trait Product<A = Self>: Sized {
    /// Method which takes an iterator and generates `Self` from the elements by
    /// multiplying the items.
    fn product<I: Iterator<Item=A>>(iter: I) -> Self;
}

macro_rules! integer_sum_product {
    ($($a:ident)*) => ($(
        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl Sum for $a {
            fn sum<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(0, |a, b| {
                    a.checked_add(b).expect("overflow in sum")
                })
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl Product for $a {
            fn product<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(1, |a, b| {
                    a.checked_mul(b).expect("overflow in product")
                })
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl<'a> Sum<&'a $a> for $a {
            fn sum<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(0, |a, b| {
                    a.checked_add(*b).expect("overflow in sum")
                })
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl<'a> Product<&'a $a> for $a {
            fn product<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(1, |a, b| {
                    a.checked_mul(*b).expect("overflow in product")
                })
            }
        }
    )*)
}

macro_rules! float_sum_product {
    ($($a:ident)*) => ($(
        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl Sum for $a {
            fn sum<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(0.0, |a, b| a + b)
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl Product for $a {
            fn product<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(1.0, |a, b| a * b)
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl<'a> Sum<&'a $a> for $a {
            fn sum<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(0.0, |a, b| a + *b)
            }
        }

        #[unstable(feature = "iter_arith_traits", issue = "34529")]
        impl<'a> Product<&'a $a> for $a {
            fn product<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(1.0, |a, b| a * *b)
            }
        }
    )*)
}

integer_sum_product! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }
float_sum_product! { f32 f64 }
