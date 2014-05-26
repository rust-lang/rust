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
//! * `Eq`
//! * `TotalEq`
//! * `Ord`
//! * `TotalOrd`
//! * `Default`
//!
//! # Examples
//!
//! Using methods:
//!
//! ```
//! let pair = ("pi", 3.14);
//! assert_eq!(pair.val0(), "pi");
//! assert_eq!(pair.val1(), 3.14);
//! ```
//!
//! Using traits implemented for tuples:
//!
//! ```
//! use std::default::Default;
//!
//! let a = (1, 2);
//! let b = (3, 4);
//! assert!(a != b);
//!
//! let c = b.clone();
//! assert!(b == c);
//!
//! let d : (u32, f32) = Default::default();
//! assert_eq!(d, (0u32, 0.0f32));
//! ```

use clone::Clone;
#[cfg(not(test))] use cmp::*;
#[cfg(not(test))] use default::Default;

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

            impl<$($T:Clone),+> Clone for ($($T,)+) {
                fn clone(&self) -> ($($T,)+) {
                    ($(self.$refN().clone(),)+)
                }
            }

            #[cfg(not(test))]
            impl<$($T:Eq),+> Eq for ($($T,)+) {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $(*self.$refN() == *other.$refN())&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $(*self.$refN() != *other.$refN())||+
                }
            }

            #[cfg(not(test))]
            impl<$($T:TotalEq),+> TotalEq for ($($T,)+) {}

            #[cfg(not(test))]
            impl<$($T:Ord + Eq),+> Ord for ($($T,)+) {
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

            #[cfg(not(test))]
            impl<$($T:TotalOrd),+> TotalOrd for ($($T,)+) {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($(self.$refN(), other.$refN()),+)
                }
            }

            #[cfg(not(test))]
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

#[cfg(test)]
mod tests {
    use super::*;
    use clone::Clone;
    use cmp::*;
    use realstd::str::{Str, StrAllocating};

    #[test]
    fn test_clone() {
        let a = (1, "2");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_getters() {
        macro_rules! test_getter(
            ($x:expr, $valN:ident, $refN:ident, $mutN:ident,
             $init:expr, $incr:expr, $result:expr) => ({
                assert_eq!($x.$valN(), $init);
                assert_eq!(*$x.$refN(), $init);
                *$x.$mutN() += $incr;
                assert_eq!(*$x.$refN(), $result);
            })
        )
        let mut x = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
        test_getter!(x, val0,  ref0,  mut0,  0,    1,   1);
        test_getter!(x, val1,  ref1,  mut1,  1,    1,   2);
        test_getter!(x, val2,  ref2,  mut2,  2,    1,   3);
        test_getter!(x, val3,  ref3,  mut3,  3,    1,   4);
        test_getter!(x, val4,  ref4,  mut4,  4,    1,   5);
        test_getter!(x, val5,  ref5,  mut5,  5,    1,   6);
        test_getter!(x, val6,  ref6,  mut6,  6,    1,   7);
        test_getter!(x, val7,  ref7,  mut7,  7,    1,   8);
        test_getter!(x, val8,  ref8,  mut8,  8,    1,   9);
        test_getter!(x, val9,  ref9,  mut9,  9,    1,   10);
        test_getter!(x, val10, ref10, mut10, 10.0, 1.0, 11.0);
        test_getter!(x, val11, ref11, mut11, 11.0, 1.0, 12.0);
    }

    #[test]
    fn test_tuple_cmp() {
        let (small, big) = ((1u, 2u, 3u), (3u, 2u, 1u));

        let nan = 0.0/0.0;

        // Eq
        assert_eq!(small, small);
        assert_eq!(big, big);
        assert!(small != big);
        assert!(big != small);

        // Ord
        assert!(small < big);
        assert!(!(small < small));
        assert!(!(big < small));
        assert!(!(big < big));

        assert!(small <= small);
        assert!(big <= big);

        assert!(big > small);
        assert!(small >= small);
        assert!(big >= small);
        assert!(big >= big);

        assert!(!((1.0, 2.0) < (nan, 3.0)));
        assert!(!((1.0, 2.0) <= (nan, 3.0)));
        assert!(!((1.0, 2.0) > (nan, 3.0)));
        assert!(!((1.0, 2.0) >= (nan, 3.0)));
        assert!(((1.0, 2.0) < (2.0, nan)));
        assert!(!((2.0, 2.0) < (2.0, nan)));

        // TotalOrd
        assert!(small.cmp(&small) == Equal);
        assert!(big.cmp(&big) == Equal);
        assert!(small.cmp(&big) == Less);
        assert!(big.cmp(&small) == Greater);
    }

    #[test]
    fn test_show() {
        let s = format!("{}", (1,));
        assert_eq!(s.as_slice(), "(1,)");
        let s = format!("{}", (1, true));
        assert_eq!(s.as_slice(), "(1, true)");
        let s = format!("{}", (1, "hi", true));
        assert_eq!(s.as_slice(), "(1, hi, true)");
    }
}
