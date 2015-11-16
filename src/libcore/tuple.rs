// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A finite heterogeneous sequence, `(T, U, ..)`
//!
//! To access a single element of a tuple one can use the `.0`
//! field access syntax.
//!
//! Indexing starts from zero, so `.0` returns first value, `.1`
//! returns second value, and so on. In general, a tuple with *N*
//! elements has field accessors from 0 to *N* - 1.
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

use clone::Clone;
use cmp::*;
use cmp::Ordering::*;
use default::Default;
use option::Option;
use option::Option::Some;

// FIXME(#19630) Remove this work-around
macro_rules! e {
    ($e:expr) => { $e }
}

// macro for implementing n-ary tuple functions and operations
macro_rules! tuple_impls {
    ($(
        $Tuple:ident {
            $(($idx:tt) -> $T:ident)+
        }
    )+) => {
        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Clone),+> Clone for ($($T,)+) {
                fn clone(&self) -> ($($T,)+) {
                    ($(e!(self.$idx.clone()),)+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:PartialEq),+> PartialEq for ($($T,)+) {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    e!($(self.$idx == other.$idx)&&+)
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    e!($(self.$idx != other.$idx)||+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Eq),+> Eq for ($($T,)+) {}

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:PartialOrd + PartialEq),+> PartialOrd for ($($T,)+) {
                #[inline]
                fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                    lexical_partial_cmp!($(self.$idx, other.$idx),+)
                }
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, $(self.$idx, other.$idx),+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Ord),+> Ord for ($($T,)+) {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($(self.$idx, other.$idx),+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
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
        if $a != $b { lexical_ord!($rel, $a, $b) }
        else { lexical_ord!($rel, $($rest_a, $rest_b),+) }
    };
    ($rel: ident, $a:expr, $b:expr) => { ($a) . $rel (& $b) };
}

macro_rules! lexical_partial_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).partial_cmp(&$b) {
            Some(Equal) => lexical_partial_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).partial_cmp(&$b) };
}

macro_rules! lexical_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).cmp(&$b) {
            Equal => lexical_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).cmp(&$b) };
}

tuple_impls! {
    Tuple1 {
        (0) -> A
    }
    Tuple2 {
        (0) -> A
        (1) -> B
    }
    Tuple3 {
        (0) -> A
        (1) -> B
        (2) -> C
    }
    Tuple4 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
    }
    Tuple5 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
    }
    Tuple6 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
    }
    Tuple7 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
    }
    Tuple8 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
    }
    Tuple9 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
    }
    Tuple10 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
    }
    Tuple11 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
    }
    Tuple12 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
        (11) -> L
    }
}
