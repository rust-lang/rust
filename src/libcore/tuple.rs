// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on tuples
//!
//! To access a single element of a tuple one can use the following
//! methods:
//!
//! * `valN` - returns a value of _N_-th element
//! * `refN` - returns a reference to _N_-th element
//! * `mutN` - returns a mutable reference to _N_-th element
//!
//! Indexing starts from zero, so `val0` returns first value, `val1`
//! returns second value, and so on. In general, a tuple with _S_
//! elements provides aforementioned methods suffixed with numbers
//! from `0` to `S-1`. Traits which contain these methods are
//! implemented for tuples with up to 12 elements.
//!
//! If every type inside a tuple implements one of the following
//! traits, then a tuple itself also implements it.
//!
//! * `Clone`
//! * `PartialEq`
//! * `Eq`
//! * `PartialOrd`
//! * `Ord`
//! * `Default`
//!
//! # Examples
//!
//! Using methods:
//!
//! ```
//! let pair = ("pi", 3.14f64);
//! assert_eq!(pair.val0(), "pi");
//! assert_eq!(pair.val1(), 3.14f64);
//! ```
//!
//! Using traits implemented for tuples:
//!
//! ```
//! use std::default::Default;
//!
//! let a = (1i, 2i);
//! let b = (3i, 4i);
//! assert!(a != b);
//!
//! let c = b.clone();
//! assert!(b == c);
//!
//! let d : (u32, f32) = Default::default();
//! assert_eq!(d, (0u32, 0.0f32));
//! ```

#![doc(primitive = "tuple")]

use clone::Clone;
use cmp::*;
use default::Default;
use option::{Option, Some};

// macro for implementing n-ary tuple functions and operations
macro_rules! tuple_impls {
    ($(
        $Tuple:ident {
            $(($valN:ident, $refN:ident, $mutN:ident) -> $T:ident {
                ($($x:ident),+) => $ret:expr
            })+
        }
    )+) => {
        $(
            #[allow(missing_doc)]
            pub trait $Tuple<$($T),+> {
                $(fn $valN(self) -> $T;)+
                $(fn $refN<'a>(&'a self) -> &'a $T;)+
                $(fn $mutN<'a>(&'a mut self) -> &'a mut $T;)+
            }

            impl<$($T),+> $Tuple<$($T),+> for ($($T,)+) {
                $(
                    #[inline]
                    #[allow(unused_variable)]
                    fn $valN(self) -> $T {
                        let ($($x,)+) = self; $ret
                    }

                    #[inline]
                    #[allow(unused_variable)]
                    fn $refN<'a>(&'a self) -> &'a $T {
                        let ($(ref $x,)+) = *self; $ret
                    }

                    #[inline]
                    #[allow(unused_variable)]
                    fn $mutN<'a>(&'a mut self) -> &'a mut $T {
                        let ($(ref mut $x,)+) = *self; $ret
                    }
                )+
            }

            #[unstable]
            impl<$($T:Clone),+> Clone for ($($T,)+) {
                fn clone(&self) -> ($($T,)+) {
                    ($(self.$refN().clone(),)+)
                }
            }

            impl<$($T:PartialEq),+> PartialEq for ($($T,)+) {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $(*self.$refN() == *other.$refN())&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $(*self.$refN() != *other.$refN())||+
                }
            }

            impl<$($T:Eq),+> Eq for ($($T,)+) {}

            impl<$($T:PartialOrd + PartialEq),+> PartialOrd for ($($T,)+) {
                #[inline]
                fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                    lexical_partial_cmp!($(self.$refN(), other.$refN()),+)
                }
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, $(self.$refN(), other.$refN()),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, $(self.$refN(), other.$refN()),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, $(self.$refN(), other.$refN()),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, $(self.$refN(), other.$refN()),+)
                }
            }

            impl<$($T:Ord),+> Ord for ($($T,)+) {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($(self.$refN(), other.$refN()),+)
                }
            }

            impl<$($T:Default),+> Default for ($($T,)+) {
                #[inline]
                fn default() -> ($($T,)+) {
                    ($({ let x: $T = Default::default(); x},)+)
                }
            }
        )+
    }
}

// Constructs an expression that performs a lexical ordering using method $rel.
// The values are interleaved, so the macro invocation for
// `(a1, a2, a3) < (b1, b2, b3)` would be `lexical_ord!(lt, a1, b1, a2, b2,
// a3, b3)` (and similarly for `lexical_cmp`)
macro_rules! lexical_ord {
    ($rel: ident, $a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        if *$a != *$b { lexical_ord!($rel, $a, $b) }
        else { lexical_ord!($rel, $($rest_a, $rest_b),+) }
    };
    ($rel: ident, $a:expr, $b:expr) => { (*$a) . $rel ($b) };
}

macro_rules! lexical_partial_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).partial_cmp($b) {
            Some(Equal) => lexical_partial_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).partial_cmp($b) };
}

macro_rules! lexical_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).cmp($b) {
            Equal => lexical_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).cmp($b) };
}

tuple_impls! {
    Tuple1 {
        (val0, ref0, mut0) -> A { (a) => a }
    }
    Tuple2 {
        (val0, ref0, mut0) -> A { (a, b) => a }
        (val1, ref1, mut1) -> B { (a, b) => b }
    }
    Tuple3 {
        (val0, ref0, mut0) -> A { (a, b, c) => a }
        (val1, ref1, mut1) -> B { (a, b, c) => b }
        (val2, ref2, mut2) -> C { (a, b, c) => c }
    }
    Tuple4 {
        (val0, ref0, mut0) -> A { (a, b, c, d) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d) => d }
    }
    Tuple5 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e) => e }
    }
    Tuple6 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e, f) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e, f) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e, f) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e, f) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e, f) => e }
        (val5, ref5, mut5) -> F { (a, b, c, d, e, f) => f }
    }
    Tuple7 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e, f, g) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e, f, g) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e, f, g) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e, f, g) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e, f, g) => e }
        (val5, ref5, mut5) -> F { (a, b, c, d, e, f, g) => f }
        (val6, ref6, mut6) -> G { (a, b, c, d, e, f, g) => g }
    }
    Tuple8 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e, f, g, h) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e, f, g, h) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e, f, g, h) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e, f, g, h) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e, f, g, h) => e }
        (val5, ref5, mut5) -> F { (a, b, c, d, e, f, g, h) => f }
        (val6, ref6, mut6) -> G { (a, b, c, d, e, f, g, h) => g }
        (val7, ref7, mut7) -> H { (a, b, c, d, e, f, g, h) => h }
    }
    Tuple9 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e, f, g, h, i) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e, f, g, h, i) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e, f, g, h, i) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e, f, g, h, i) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e, f, g, h, i) => e }
        (val5, ref5, mut5) -> F { (a, b, c, d, e, f, g, h, i) => f }
        (val6, ref6, mut6) -> G { (a, b, c, d, e, f, g, h, i) => g }
        (val7, ref7, mut7) -> H { (a, b, c, d, e, f, g, h, i) => h }
        (val8, ref8, mut8) -> I { (a, b, c, d, e, f, g, h, i) => i }
    }
    Tuple10 {
        (val0, ref0, mut0) -> A { (a, b, c, d, e, f, g, h, i, j) => a }
        (val1, ref1, mut1) -> B { (a, b, c, d, e, f, g, h, i, j) => b }
        (val2, ref2, mut2) -> C { (a, b, c, d, e, f, g, h, i, j) => c }
        (val3, ref3, mut3) -> D { (a, b, c, d, e, f, g, h, i, j) => d }
        (val4, ref4, mut4) -> E { (a, b, c, d, e, f, g, h, i, j) => e }
        (val5, ref5, mut5) -> F { (a, b, c, d, e, f, g, h, i, j) => f }
        (val6, ref6, mut6) -> G { (a, b, c, d, e, f, g, h, i, j) => g }
        (val7, ref7, mut7) -> H { (a, b, c, d, e, f, g, h, i, j) => h }
        (val8, ref8, mut8) -> I { (a, b, c, d, e, f, g, h, i, j) => i }
        (val9, ref9, mut9) -> J { (a, b, c, d, e, f, g, h, i, j) => j }
    }
    Tuple11 {
        (val0,  ref0,  mut0)  -> A { (a, b, c, d, e, f, g, h, i, j, k) => a }
        (val1,  ref1,  mut1)  -> B { (a, b, c, d, e, f, g, h, i, j, k) => b }
        (val2,  ref2,  mut2)  -> C { (a, b, c, d, e, f, g, h, i, j, k) => c }
        (val3,  ref3,  mut3)  -> D { (a, b, c, d, e, f, g, h, i, j, k) => d }
        (val4,  ref4,  mut4)  -> E { (a, b, c, d, e, f, g, h, i, j, k) => e }
        (val5,  ref5,  mut5)  -> F { (a, b, c, d, e, f, g, h, i, j, k) => f }
        (val6,  ref6,  mut6)  -> G { (a, b, c, d, e, f, g, h, i, j, k) => g }
        (val7,  ref7,  mut7)  -> H { (a, b, c, d, e, f, g, h, i, j, k) => h }
        (val8,  ref8,  mut8)  -> I { (a, b, c, d, e, f, g, h, i, j, k) => i }
        (val9,  ref9,  mut9)  -> J { (a, b, c, d, e, f, g, h, i, j, k) => j }
        (val10, ref10, mut10) -> K { (a, b, c, d, e, f, g, h, i, j, k) => k }
    }
    Tuple12 {
        (val0,  ref0,  mut0)  -> A { (a, b, c, d, e, f, g, h, i, j, k, l) => a }
        (val1,  ref1,  mut1)  -> B { (a, b, c, d, e, f, g, h, i, j, k, l) => b }
        (val2,  ref2,  mut2)  -> C { (a, b, c, d, e, f, g, h, i, j, k, l) => c }
        (val3,  ref3,  mut3)  -> D { (a, b, c, d, e, f, g, h, i, j, k, l) => d }
        (val4,  ref4,  mut4)  -> E { (a, b, c, d, e, f, g, h, i, j, k, l) => e }
        (val5,  ref5,  mut5)  -> F { (a, b, c, d, e, f, g, h, i, j, k, l) => f }
        (val6,  ref6,  mut6)  -> G { (a, b, c, d, e, f, g, h, i, j, k, l) => g }
        (val7,  ref7,  mut7)  -> H { (a, b, c, d, e, f, g, h, i, j, k, l) => h }
        (val8,  ref8,  mut8)  -> I { (a, b, c, d, e, f, g, h, i, j, k, l) => i }
        (val9,  ref9,  mut9)  -> J { (a, b, c, d, e, f, g, h, i, j, k, l) => j }
        (val10, ref10, mut10) -> K { (a, b, c, d, e, f, g, h, i, j, k, l) => k }
        (val11, ref11, mut11) -> L { (a, b, c, d, e, f, g, h, i, j, k, l) => l }
    }
}

