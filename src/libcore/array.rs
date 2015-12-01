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
//!
//! *[See also the array primitive type](../primitive.array.html).*

#![unstable(feature = "fixed_size_array",
            reason = "traits and impls are better expressed through generic \
                      integer constants",
            issue = "27778")]

use borrow::{Borrow, BorrowMut};
use clone::Clone;
use cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use convert::{AsRef, AsMut};
use default::Default;
use fmt;
use hash::{Hash, self};
use iter::IntoIterator;
use marker::{Sized, Unsize};
use option::Option;
use slice::{Iter, IterMut, SliceExt};

/// Utility trait implemented only on arrays of fixed size
///
/// This trait can be used to implement other traits on fixed-size arrays
/// without causing much metadata bloat.
///
/// The trait is marked unsafe in order to restrict implementors to fixed-size
/// arrays. User of this trait can assume that implementors have the exact
/// layout in memory of a fixed size array (for example, for unsafe
/// initialization).
///
/// Note that the traits AsRef and AsMut provide similar methods for types that
/// may not be fixed-size arrays. Implementors should prefer those traits
/// instead.
pub unsafe trait FixedSizeArray<T> {
    /// Converts the array to immutable slice
    fn as_slice(&self) -> &[T];
    /// Converts the array to mutable slice
    fn as_mut_slice(&mut self) -> &mut [T];
}

unsafe impl<T, A: Unsize<[T]>> FixedSizeArray<T> for A {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

// macro for implementing n-ary tuple functions and operations
macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            impl<T> AsRef<[T]> for [T; $N] {
                #[inline]
                fn as_ref(&self) -> &[T] {
                    &self[..]
                }
            }

            impl<T> AsMut<[T]> for [T; $N] {
                #[inline]
                fn as_mut(&mut self) -> &mut [T] {
                    &mut self[..]
                }
            }

            #[stable(feature = "array_borrow", since = "1.4.0")]
            impl<T> Borrow<[T]> for [T; $N] {
                fn borrow(&self) -> &[T] {
                    self
                }
            }

            #[stable(feature = "array_borrow", since = "1.4.0")]
            impl<T> BorrowMut<[T]> for [T; $N] {
                fn borrow_mut(&mut self) -> &mut [T] {
                    self
                }
            }

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

            // NOTE: some less important impls are omitted to reduce code bloat
            __impl_slice_eq1! { [A; $N], [B; $N] }
            __impl_slice_eq2! { [A; $N], [B] }
            __impl_slice_eq2! { [A; $N], &'b [B] }
            __impl_slice_eq2! { [A; $N], &'b mut [B] }
            // __impl_slice_eq2! { [A; $N], &'b [B; $N] }
            // __impl_slice_eq2! { [A; $N], &'b mut [B; $N] }

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

// The Default impls cannot be generated using the array_impls! macro because
// they require array literals.

macro_rules! array_impl_default {
    {$n:expr, $t:ident $($ts:ident)*} => {
        #[stable(since = "1.4.0", feature = "array_default")]
        impl<T> Default for [T; $n] where T: Default {
            fn default() -> [T; $n] {
                [$t::default(), $($ts::default()),*]
            }
        }
        array_impl_default!{($n - 1), $($ts)*}
    };
    {$n:expr,} => {
        #[stable(since = "1.4.0", feature = "array_default")]
        impl<T> Default for [T; $n] {
            fn default() -> [T; $n] { [] }
        }
    };
}

array_impl_default!{32, T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T}

macro_rules! array_impl_clone {
    {$n:expr, $t:ident $($ts:ident)*} => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T: Clone> Clone for [T; $n] {
            fn clone(&self) -> [T; $n] {
                let &[ref $t, $(ref $ts),*] = self;
                [$t.clone(), $($ts.clone()),*]
            }
        }
        array_impl_clone!{($n - 1), $($ts)*}
    };
    {$n:expr,} => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T: Clone> Clone for [T; 0] {
            fn clone(&self) -> [T; 0] {
                []
            }
        }
    };
}

array_impl_clone! { 32,
    t00 t01 t02 t03 t04 t05 t06 t07 t08 t09
    t10 t11 t12 t13 t14 t15 t16 t17 t18 t19
    t20 t21 t22 t23 t24 t25 t26 t27 t28 t29
    t30 t31
}
