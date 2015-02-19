// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of things like `Eq` for fixed-length arrays
//! up to a certain length. Eventually we should able to generalize
//! to all lengths.

#![unstable(feature = "core")] // not yet reviewed

use clone::Clone;
use cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use fmt;
use hash::{Hash, self};
use iter::IntoIterator;
use marker::Copy;
use ops::Deref;
use option::Option;
use slice::{Iter, IterMut, SliceExt};

// macro for implementing n-ary tuple functions and operations
macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T:Copy> Clone for [T; $N] {
                fn clone(&self) -> [T; $N] {
                    *self
                }
            }

            #[cfg(stage0)]
            impl<S: hash::Writer + hash::Hasher, T: Hash<S>> Hash<S> for [T; $N] {
                fn hash(&self, state: &mut S) {
                    Hash::hash(&self[..], state)
                }
            }
            #[cfg(not(stage0))]
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T: Hash> Hash for [T; $N] {
                fn hash<H: hash::Hasher>(&self, state: &mut H) {
                    Hash::hash(&self[..], state)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T: fmt::Debug> fmt::Debug for [T; $N] {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Debug::fmt(&&self[..], f)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<'a, T> IntoIterator for &'a [T; $N] {
                type Item = &'a T;
                type IntoIter = Iter<'a, T>;

                fn into_iter(self) -> Iter<'a, T> {
                    self.iter()
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<'a, T> IntoIterator for &'a mut [T; $N] {
                type Item = &'a mut T;
                type IntoIter = IterMut<'a, T>;

                fn into_iter(self) -> IterMut<'a, T> {
                    self.iter_mut()
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<A, B> PartialEq<[B; $N]> for [A; $N] where A: PartialEq<B> {
                #[inline]
                fn eq(&self, other: &[B; $N]) -> bool {
                    &self[..] == &other[..]
                }
                #[inline]
                fn ne(&self, other: &[B; $N]) -> bool {
                    &self[..] != &other[..]
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<'a, A, B, Rhs> PartialEq<Rhs> for [A; $N] where
                A: PartialEq<B>,
                Rhs: Deref<Target=[B]>,
            {
                #[inline(always)]
                fn eq(&self, other: &Rhs) -> bool {
                    PartialEq::eq(&self[..], &**other)
                }
                #[inline(always)]
                fn ne(&self, other: &Rhs) -> bool {
                    PartialEq::ne(&self[..], &**other)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<'a, A, B, Lhs> PartialEq<[B; $N]> for Lhs where
                A: PartialEq<B>,
                Lhs: Deref<Target=[A]>
            {
                #[inline(always)]
                fn eq(&self, other: &[B; $N]) -> bool {
                    PartialEq::eq(&**self, &other[..])
                }
                #[inline(always)]
                fn ne(&self, other: &[B; $N]) -> bool {
                    PartialEq::ne(&**self, &other[..])
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T:Eq> Eq for [T; $N] { }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T:PartialOrd> PartialOrd for [T; $N] {
                #[inline]
                fn partial_cmp(&self, other: &[T; $N]) -> Option<Ordering> {
                    PartialOrd::partial_cmp(&&self[..], &&other[..])
                }
                #[inline]
                fn lt(&self, other: &[T; $N]) -> bool {
                    PartialOrd::lt(&&self[..], &&other[..])
                }
                #[inline]
                fn le(&self, other: &[T; $N]) -> bool {
                    PartialOrd::le(&&self[..], &&other[..])
                }
                #[inline]
                fn ge(&self, other: &[T; $N]) -> bool {
                    PartialOrd::ge(&&self[..], &&other[..])
                }
                #[inline]
                fn gt(&self, other: &[T; $N]) -> bool {
                    PartialOrd::gt(&&self[..], &&other[..])
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<T:Ord> Ord for [T; $N] {
                #[inline]
                fn cmp(&self, other: &[T; $N]) -> Ordering {
                    Ord::cmp(&&self[..], &&other[..])
                }
            }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}
