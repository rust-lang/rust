// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use ops::{Mul, Add};
use num::Wrapping;

/// Conversion from an `Iterator`.
///
/// By implementing `FromIterator` for a type, you define how it will be
/// created from an iterator. This is common for types which describe a
/// collection of some kind.
///
/// `FromIterator`'s [`from_iter`] is rarely called explicitly, and is instead
/// used through [`Iterator`]'s [`collect`] method. See [`collect`]'s
/// documentation for more examples.
///
/// [`from_iter`]: #tymethod.from_iter
/// [`Iterator`]: trait.Iterator.html
/// [`collect`]: trait.Iterator.html#method.collect
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
/// Using [`collect`] to implicitly use `FromIterator`:
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
/// to extend a collection by including the contents of that iterator. When
/// extending a collection with an already existing key, that entry is updated
/// or, in the case of collections that permit multiple entries with equal
/// keys, that entry is inserted.
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
    /// Removes and returns an element from the end of the iterator.
    ///
    /// Returns `None` when there are no more elements.
    ///
    /// The [trait-level] docs contain more details.
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

    /// Searches for an element of an iterator from the right that satisfies a predicate.
    ///
    /// `rfind()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, starting at the end, and if any
    /// of them return `true`, then `rfind()` returns [`Some(element)`]. If they all return
    /// `false`, it returns [`None`].
    ///
    /// `rfind()` is short-circuiting; in other words, it will stop processing
    /// as soon as the closure returns `true`.
    ///
    /// Because `rfind()` takes a reference, and many iterators iterate over
    /// references, this leads to a possibly confusing situation where the
    /// argument is a double reference. You can see this effect in the
    /// examples below, with `&&x`.
    ///
    /// [`Some(element)`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_rfind)]
    ///
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.iter().rfind(|&&x| x == 2), Some(&2));
    ///
    /// assert_eq!(a.iter().rfind(|&&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// #![feature(iter_rfind)]
    ///
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.rfind(|&&x| x == 2), Some(&2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next_back(), Some(&1));
    /// ```
    #[inline]
    #[unstable(feature = "iter_rfind", issue = "39480")]
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item> where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool
    {
        for x in self.by_ref().rev() {
            if predicate(&x) { return Some(x) }
        }
        None
    }
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
/// [`Iterator`]. When doing so, the implementation of [`size_hint`] *must*
/// return the exact size of the iterator.
///
/// [`Iterator`]: trait.Iterator.html
/// [`size_hint`]: trait.Iterator.html#method.size_hint
///
/// The [`len`] method has a default implementation, so you usually shouldn't
/// implement it. However, you may be able to provide a more performant
/// implementation than the default, so overriding it in this case makes sense.
///
/// [`len`]: #method.len
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
///     // We can easily calculate the remaining number of iterations.
///     fn len(&self) -> usize {
///         5 - self.count
///     }
/// }
///
/// // And now we can use it!
///
/// let counter = Counter::new();
///
/// assert_eq!(5, counter.len());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExactSizeIterator: Iterator {
    /// Returns the exact number of times the iterator will iterate.
    ///
    /// This method has a default implementation, so you usually should not
    /// implement it directly. However, if you can provide a more efficient
    /// implementation, you can do so. See the [trait-level] docs for an
    /// example.
    ///
    /// This function has the same safety guarantees as the [`size_hint`]
    /// function.
    ///
    /// [trait-level]: trait.ExactSizeIterator.html
    /// [`size_hint`]: trait.Iterator.html#method.size_hint
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }

    /// Returns whether the iterator is empty.
    ///
    /// This method has a default implementation using `self.len()`, so you
    /// don't need to implement it yourself.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(exact_size_is_empty)]
    ///
    /// let mut one_element = 0..1;
    /// assert!(!one_element.is_empty());
    ///
    /// assert_eq!(one_element.next(), Some(0));
    /// assert!(one_element.is_empty());
    ///
    /// assert_eq!(one_element.next(), None);
    /// ```
    #[inline]
    #[unstable(feature = "exact_size_is_empty", issue = "35428")]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: ExactSizeIterator + ?Sized> ExactSizeIterator for &'a mut I {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

/// Trait to represent types that can be created by summing up an iterator.
///
/// This trait is used to implement the [`sum`] method on iterators. Types which
/// implement the trait can be generated by the [`sum`] method. Like
/// [`FromIterator`] this trait should rarely be called directly and instead
/// interacted with through [`Iterator::sum`].
///
/// [`sum`]: ../../std/iter/trait.Sum.html#tymethod.sum
/// [`FromIterator`]: ../../std/iter/trait.FromIterator.html
/// [`Iterator::sum`]: ../../std/iter/trait.Iterator.html#method.sum
#[stable(feature = "iter_arith_traits", since = "1.12.0")]
pub trait Sum<A = Self>: Sized {
    /// Method which takes an iterator and generates `Self` from the elements by
    /// "summing up" the items.
    #[stable(feature = "iter_arith_traits", since = "1.12.0")]
    fn sum<I: Iterator<Item=A>>(iter: I) -> Self;
}

/// Trait to represent types that can be created by multiplying elements of an
/// iterator.
///
/// This trait is used to implement the [`product`] method on iterators. Types
/// which implement the trait can be generated by the [`product`] method. Like
/// [`FromIterator`] this trait should rarely be called directly and instead
/// interacted with through [`Iterator::product`].
///
/// [`product`]: ../../std/iter/trait.Product.html#tymethod.product
/// [`FromIterator`]: ../../std/iter/trait.FromIterator.html
/// [`Iterator::product`]: ../../std/iter/trait.Iterator.html#method.product
#[stable(feature = "iter_arith_traits", since = "1.12.0")]
pub trait Product<A = Self>: Sized {
    /// Method which takes an iterator and generates `Self` from the elements by
    /// multiplying the items.
    #[stable(feature = "iter_arith_traits", since = "1.12.0")]
    fn product<I: Iterator<Item=A>>(iter: I) -> Self;
}

// NB: explicitly use Add and Mul here to inherit overflow checks
macro_rules! integer_sum_product {
    (@impls $zero:expr, $one:expr, #[$attr:meta], $($a:ty)*) => ($(
        #[$attr]
        impl Sum for $a {
            fn sum<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold($zero, Add::add)
            }
        }

        #[$attr]
        impl Product for $a {
            fn product<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold($one, Mul::mul)
            }
        }

        #[$attr]
        impl<'a> Sum<&'a $a> for $a {
            fn sum<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold($zero, Add::add)
            }
        }

        #[$attr]
        impl<'a> Product<&'a $a> for $a {
            fn product<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold($one, Mul::mul)
            }
        }
    )*);
    ($($a:ty)*) => (
        integer_sum_product!(@impls 0, 1,
                #[stable(feature = "iter_arith_traits", since = "1.12.0")],
                $($a)+);
        integer_sum_product!(@impls Wrapping(0), Wrapping(1),
                #[stable(feature = "wrapping_iter_arith", since = "1.14.0")],
                $(Wrapping<$a>)+);
    );
}

macro_rules! float_sum_product {
    ($($a:ident)*) => ($(
        #[stable(feature = "iter_arith_traits", since = "1.12.0")]
        impl Sum for $a {
            fn sum<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(0.0, |a, b| a + b)
            }
        }

        #[stable(feature = "iter_arith_traits", since = "1.12.0")]
        impl Product for $a {
            fn product<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold(1.0, |a, b| a * b)
            }
        }

        #[stable(feature = "iter_arith_traits", since = "1.12.0")]
        impl<'a> Sum<&'a $a> for $a {
            fn sum<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(0.0, |a, b| a + *b)
            }
        }

        #[stable(feature = "iter_arith_traits", since = "1.12.0")]
        impl<'a> Product<&'a $a> for $a {
            fn product<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold(1.0, |a, b| a * *b)
            }
        }
    )*)
}

integer_sum_product! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }
float_sum_product! { f32 f64 }

/// An iterator adapter that produces output as long as the underlying
/// iterator produces `Result::Ok` values.
///
/// If an error is encountered, the iterator stops and the error is
/// stored. The error may be recovered later via `reconstruct`.
struct ResultShunt<I, E> {
    iter: I,
    error: Option<E>,
}

impl<I, T, E> ResultShunt<I, E>
    where I: Iterator<Item = Result<T, E>>
{
    /// Process the given iterator as if it yielded a `T` instead of a
    /// `Result<T, _>`. Any errors will stop the inner iterator and
    /// the overall result will be an error.
    pub fn process<F, U>(iter: I, mut f: F) -> Result<U, E>
        where F: FnMut(&mut Self) -> U
    {
        let mut shunt = ResultShunt::new(iter);
        let value = f(shunt.by_ref());
        shunt.reconstruct(value)
    }

    fn new(iter: I) -> Self {
        ResultShunt {
            iter: iter,
            error: None,
        }
    }

    /// Consume the adapter and rebuild a `Result` value. This should
    /// *always* be called, otherwise any potential error would be
    /// lost.
    fn reconstruct<U>(self, val: U) -> Result<U, E> {
        match self.error {
            None => Ok(val),
            Some(e) => Err(e),
        }
    }
}

impl<I, T, E> Iterator for ResultShunt<I, E>
    where I: Iterator<Item = Result<T, E>>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => {
                self.error = Some(e);
                None
            }
            None => None,
        }
    }
}

#[stable(feature = "iter_arith_traits_result", since="1.16.0")]
impl<T, U, E> Sum<Result<U, E>> for Result<T, E>
    where T: Sum<U>,
{
    fn sum<I>(iter: I) -> Result<T, E>
        where I: Iterator<Item = Result<U, E>>,
    {
        ResultShunt::process(iter, |i| i.sum())
    }
}

#[stable(feature = "iter_arith_traits_result", since="1.16.0")]
impl<T, U, E> Product<Result<U, E>> for Result<T, E>
    where T: Product<U>,
{
    fn product<I>(iter: I) -> Result<T, E>
        where I: Iterator<Item = Result<U, E>>,
    {
        ResultShunt::process(iter, |i| i.product())
    }
}

/// An iterator that always continues to yield `None` when exhausted.
///
/// Calling next on a fused iterator that has returned `None` once is guaranteed
/// to return [`None`] again. This trait is should be implemented by all iterators
/// that behave this way because it allows for some significant optimizations.
///
/// Note: In general, you should not use `FusedIterator` in generic bounds if
/// you need a fused iterator. Instead, you should just call [`Iterator::fuse`]
/// on the iterator. If the iterator is already fused, the additional [`Fuse`]
/// wrapper will be a no-op with no performance penalty.
///
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [`Iterator::fuse`]: ../../std/iter/trait.Iterator.html#method.fuse
/// [`Fuse`]: ../../std/iter/struct.Fuse.html
#[unstable(feature = "fused", issue = "35602")]
pub trait FusedIterator: Iterator {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, I: FusedIterator + ?Sized> FusedIterator for &'a mut I {}

/// An iterator that reports an accurate length using size_hint.
///
/// The iterator reports a size hint where it is either exact
/// (lower bound is equal to upper bound), or the upper bound is [`None`].
/// The upper bound must only be [`None`] if the actual iterator length is
/// larger than [`usize::MAX`].
///
/// The iterator must produce exactly the number of elements it reported.
///
/// # Safety
///
/// This trait must only be implemented when the contract is upheld.
/// Consumers of this trait must inspect [`.size_hint`]’s upper bound.
///
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [`usize::MAX`]: ../../std/usize/constant.MAX.html
/// [`.size_hint`]: ../../std/iter/trait.Iterator.html#method.size_hint
#[unstable(feature = "trusted_len", issue = "37572")]
pub unsafe trait TrustedLen : Iterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, I: TrustedLen + ?Sized> TrustedLen for &'a mut I {}
