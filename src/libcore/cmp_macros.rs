// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Utility macros for implementing PartialEq on slice-like types

#![doc(hidden)]

#[macro_export]
macro_rules! __impl_slice_eq1 {
    ($Lhs: ty, $Rhs: ty) => {
        __impl_slice_eq1! { $Lhs, $Rhs, Sized }
    };
    ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
            #[inline]
            fn eq(&self, other: &$Rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$Rhs) -> bool { self[..] != other[..] }
        }
    }
}

#[macro_export]
macro_rules! __impl_slice_eq2 {
    ($Lhs: ty, $Rhs: ty) => {
        __impl_slice_eq2! { $Lhs, $Rhs, Sized }
    };
    ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
        __impl_slice_eq1!($Lhs, $Rhs, $Bound);

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b, A: $Bound, B> PartialEq<$Lhs> for $Rhs where B: PartialEq<A> {
            #[inline]
            fn eq(&self, other: &$Lhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$Lhs) -> bool { self[..] != other[..] }
        }
    }
}
