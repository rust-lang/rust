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

#[allow(missing_doc)];

use clone::Clone;
#[cfg(not(test))] use cmp::*;
#[cfg(not(test))] use default::Default;

pub use prim::tuple::{ImmutableTuple1, ImmutableTuple2, ImmutableTuple3, ImmutableTuple4};
pub use prim::tuple::{ImmutableTuple5, ImmutableTuple6, ImmutableTuple7, ImmutableTuple8};
pub use prim::tuple::{ImmutableTuple9, ImmutableTuple10, ImmutableTuple11, ImmutableTuple12};
pub use prim::tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use prim::tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use prim::tuple::{Tuple9, Tuple10, Tuple11, Tuple12};


/// Method extensions to pairs where both types satisfy the `Clone` bound
pub trait CloneableTuple<T, U> {
    /// Return the first element of self
    fn first(&self) -> T;
    /// Return the second element of self
    fn second(&self) -> U;
    /// Return the results of swapping the two elements of self
    fn swap(&self) -> (U, T);
}

impl<T:Clone,U:Clone> CloneableTuple<T, U> for (T, U) {
    /// Return the first element of self
    #[inline]
    fn first(&self) -> T {
        match *self {
            (ref t, _) => (*t).clone(),
        }
    }

    /// Return the second element of self
    #[inline]
    fn second(&self) -> U {
        match *self {
            (_, ref u) => (*u).clone(),
        }
    }

    /// Return the results of swapping the two elements of self
    #[inline]
    fn swap(&self) -> (U, T) {
        match (*self).clone() {
            (t, u) => (u, t),
        }
    }
}

/// Method extensions for pairs where the types don't necessarily satisfy the
/// `Clone` bound
pub trait ImmutableTuple<T, U> {
    /// Return a reference to the first element of self
    fn first_ref<'a>(&'a self) -> &'a T;
    /// Return a reference to the second element of self
    fn second_ref<'a>(&'a self) -> &'a U;
}

impl<T, U> ImmutableTuple<T, U> for (T, U) {
    #[inline]
    fn first_ref<'a>(&'a self) -> &'a T {
        match *self {
            (ref t, _) => t,
        }
    }
    #[inline]
    fn second_ref<'a>(&'a self) -> &'a U {
        match *self {
            (_, ref u) => u,
        }
    }
}

// macro for implementing n-ary tuple functions and operations

macro_rules! tuple_impls {
    ($(
       {
            $(($get_ref_fn:ident) -> $T:ident;
            )+
        }
    )+) => {
        $(

            impl<$($T:Clone),+> Clone for ($($T,)+) {
                fn clone(&self) -> ($($T,)+) {
                    ($(self.$get_ref_fn().clone(),)+)
                }
            }

            #[cfg(not(test))]
            impl<$($T:Eq),+> Eq for ($($T,)+) {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $(*self.$get_ref_fn() == *other.$get_ref_fn())&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $(*self.$get_ref_fn() != *other.$get_ref_fn())||+
                }
            }

            #[cfg(not(test))]
            impl<$($T:TotalEq),+> TotalEq for ($($T,)+) {
                #[inline]
                fn equals(&self, other: &($($T,)+)) -> bool {
                    $(self.$get_ref_fn().equals(other.$get_ref_fn()))&&+
                }
            }

            #[cfg(not(test))]
            impl<$($T:Ord + Eq),+> Ord for ($($T,)+) {
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, $(self.$get_ref_fn(), other.$get_ref_fn()),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, $(self.$get_ref_fn(), other.$get_ref_fn()),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, $(self.$get_ref_fn(), other.$get_ref_fn()),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, $(self.$get_ref_fn(), other.$get_ref_fn()),+)
                }
            }

            #[cfg(not(test))]
            impl<$($T:TotalOrd),+> TotalOrd for ($($T,)+) {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($(self.$get_ref_fn(), other.$get_ref_fn()),+)
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
    {
        (n0_ref) -> A;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
        (n5_ref) -> F;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
        (n5_ref) -> F;
        (n6_ref) -> G;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
        (n5_ref) -> F;
        (n6_ref) -> G;
        (n7_ref) -> H;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
        (n5_ref) -> F;
        (n6_ref) -> G;
        (n7_ref) -> H;
        (n8_ref) -> I;
    }

    {
        (n0_ref) -> A;
        (n1_ref) -> B;
        (n2_ref) -> C;
        (n3_ref) -> D;
        (n4_ref) -> E;
        (n5_ref) -> F;
        (n6_ref) -> G;
        (n7_ref) -> H;
        (n8_ref) -> I;
        (n9_ref) -> J;
    }

    {
        (n0_ref)  -> A;
        (n1_ref)  -> B;
        (n2_ref)  -> C;
        (n3_ref)  -> D;
        (n4_ref)  -> E;
        (n5_ref)  -> F;
        (n6_ref)  -> G;
        (n7_ref)  -> H;
        (n8_ref)  -> I;
        (n9_ref)  -> J;
        (n10_ref) -> K;
    }

    {
        (n0_ref)  -> A;
        (n1_ref)  -> B;
        (n2_ref)  -> C;
        (n3_ref)  -> D;
        (n4_ref)  -> E;
        (n5_ref)  -> F;
        (n6_ref)  -> G;
        (n7_ref)  -> H;
        (n8_ref)  -> I;
        (n9_ref)  -> J;
        (n10_ref) -> K;
        (n11_ref) -> L;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clone::Clone;
    use cmp::*;

    #[test]
    fn test_tuple_ref() {
        let x = (~"foo", ~"bar");
        assert_eq!(x.first_ref(), &~"foo");
        assert_eq!(x.second_ref(), &~"bar");
    }

    #[test]
    fn test_tuple() {
        assert_eq!((948, 4039.48).first(), 948);
        assert_eq!((34.5, ~"foo").second(), ~"foo");
        assert_eq!(('a', 2).swap(), (2, 'a'));
    }

    #[test]
    fn test_clone() {
        let a = (1, ~"2");
        let b = a.clone();
        assert_eq!(a.first(), b.first());
        assert_eq!(a.second(), b.second());
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

        // TotalEq
        assert!(small.equals(&small));
        assert!(big.equals(&big));
        assert!(!small.equals(&big));
        assert!(!big.equals(&small));

        // TotalOrd
        assert_eq!(small.cmp(&small), Equal);
        assert_eq!(big.cmp(&big), Equal);
        assert_eq!(small.cmp(&big), Less);
        assert_eq!(big.cmp(&small), Greater);
    }
}
