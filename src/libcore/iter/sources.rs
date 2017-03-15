// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use marker;
use usize;

use super::{FusedIterator, TrustedLen};

/// An iterator that repeats an element endlessly.
///
/// This `struct` is created by the [`repeat`] function. See its documentation for more.
///
/// [`repeat`]: fn.repeat.html
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat<A> {
    element: A
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> Iterator for Repeat<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> { Some(self.element.clone()) }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { (usize::MAX, None) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> DoubleEndedIterator for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { Some(self.element.clone()) }
}

#[unstable(feature = "fused", issue = "35602")]
impl<A: Clone> FusedIterator for Repeat<A> {}

/// Creates a new iterator that endlessly repeats a single element.
///
/// The `repeat()` function repeats a single value over and over and over and
/// over and over and 🔁.
///
/// Infinite iterators like `repeat()` are often used with adapters like
/// [`take`], in order to make them finite.
///
/// [`take`]: trait.Iterator.html#method.take
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // the number four 4ever:
/// let mut fours = iter::repeat(4);
///
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
///
/// // yup, still four
/// assert_eq!(Some(4), fours.next());
/// ```
///
/// Going finite with [`take`]:
///
/// ```
/// use std::iter;
///
/// // that last example was too many fours. Let's only have four fours.
/// let mut four_fours = iter::repeat(4).take(4);
///
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
///
/// // ... and now we're done
/// assert_eq!(None, four_fours.next());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn repeat<T: Clone>(elt: T) -> Repeat<T> {
    Repeat{element: elt}
}

/// An iterator that yields nothing.
///
/// This `struct` is created by the [`empty`] function. See its documentation for more.
///
/// [`empty`]: fn.empty.html
#[stable(feature = "iter_empty", since = "1.2.0")]
pub struct Empty<T>(marker::PhantomData<T>);

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T> fmt::Debug for Empty<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Empty")
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Iterator for Empty<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (0, Some(0))
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> DoubleEndedIterator for Empty<T> {
    fn next_back(&mut self) -> Option<T> {
        None
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> ExactSizeIterator for Empty<T> {
    fn len(&self) -> usize {
        0
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Empty<T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<T> FusedIterator for Empty<T> {}

// not #[derive] because that adds a Clone bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Clone for Empty<T> {
    fn clone(&self) -> Empty<T> {
        Empty(marker::PhantomData)
    }
}

// not #[derive] because that adds a Default bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Default for Empty<T> {
    fn default() -> Empty<T> {
        Empty(marker::PhantomData)
    }
}

/// Creates an iterator that yields nothing.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // this could have been an iterator over i32, but alas, it's just not.
/// let mut nope = iter::empty::<i32>();
///
/// assert_eq!(None, nope.next());
/// ```
#[stable(feature = "iter_empty", since = "1.2.0")]
pub fn empty<T>() -> Empty<T> {
    Empty(marker::PhantomData)
}

/// An iterator that yields an element exactly once.
///
/// This `struct` is created by the [`once`] function. See its documentation for more.
///
/// [`once`]: fn.once.html
#[derive(Clone, Debug)]
#[stable(feature = "iter_once", since = "1.2.0")]
pub struct Once<T> {
    inner: ::option::IntoIter<T>
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> Iterator for Once<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> DoubleEndedIterator for Once<T> {
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> ExactSizeIterator for Once<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Once<T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<T> FusedIterator for Once<T> {}

/// Creates an iterator that yields an element exactly once.
///
/// This is commonly used to adapt a single value into a [`chain`] of other
/// kinds of iteration. Maybe you have an iterator that covers almost
/// everything, but you need an extra special case. Maybe you have a function
/// which works on iterators, but you only need to process one value.
///
/// [`chain`]: trait.Iterator.html#method.chain
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // one is the loneliest number
/// let mut one = iter::once(1);
///
/// assert_eq!(Some(1), one.next());
///
/// // just one, that's all we get
/// assert_eq!(None, one.next());
/// ```
///
/// Chaining together with another iterator. Let's say that we want to iterate
/// over each file of the `.foo` directory, but also a configuration file,
/// `.foorc`:
///
/// ```no_run
/// use std::iter;
/// use std::fs;
/// use std::path::PathBuf;
///
/// let dirs = fs::read_dir(".foo").unwrap();
///
/// // we need to convert from an iterator of DirEntry-s to an iterator of
/// // PathBufs, so we use map
/// let dirs = dirs.map(|file| file.unwrap().path());
///
/// // now, our iterator just for our config file
/// let config = iter::once(PathBuf::from(".foorc"));
///
/// // chain the two iterators together into one big iterator
/// let files = dirs.chain(config);
///
/// // this will give us all of the files in .foo as well as .foorc
/// for f in files {
///     println!("{:?}", f);
/// }
/// ```
#[stable(feature = "iter_once", since = "1.2.0")]
pub fn once<T>(value: T) -> Once<T> {
    Once { inner: Some(value).into_iter() }
}
