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
#[cfg(not(test))] use num::Zero;

/// Method extensions to pairs where both types satisfy the `Clone` bound
pub trait CopyableTuple<T, U> {
    /// Return the first element of self
    fn first(&self) -> T;
    /// Return the second element of self
    fn second(&self) -> U;
    /// Return the results of swapping the two elements of self
    fn swap(&self) -> (U, T);
}

impl<T:Clone,U:Clone> CopyableTuple<T, U> for (T, U) {
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
        ($move_trait:ident, $immutable_trait:ident) {
            $(($get_fn:ident, $get_ref_fn:ident) -> $T:ident {
                $move_pattern:pat, $ref_pattern:pat => $ret:expr
            })+
        }
    )+) => {
        $(
            pub trait $move_trait<$($T),+> {
                $(fn $get_fn(self) -> $T;)+
            }

            impl<$($T),+> $move_trait<$($T),+> for ($($T,)+) {
                $(
                    #[inline]
                    fn $get_fn(self) -> $T {
                        let $move_pattern = self;
                        $ret
                    }
                )+
            }

            pub trait $immutable_trait<$($T),+> {
                $(fn $get_ref_fn<'a>(&'a self) -> &'a $T;)+
            }

            impl<$($T),+> $immutable_trait<$($T),+> for ($($T,)+) {
                $(
                    #[inline]
                    fn $get_ref_fn<'a>(&'a self) -> &'a $T {
                        let $ref_pattern = *self;
                        $ret
                    }
                )+
            }

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

            #[cfg(not(test))]
            impl<$($T:Zero),+> Zero for ($($T,)+) {
                #[inline]
                fn zero() -> ($($T,)+) {
                    ($({ let x: $T = Zero::zero(); x},)+)
                }
                #[inline]
                fn is_zero(&self) -> bool {
                    $(self.$get_ref_fn().is_zero())&&+
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
    (Tuple1, ImmutableTuple1) {
        (n0, n0_ref) -> A { (a,), (ref a,) => a }
    }

    (Tuple2, ImmutableTuple2) {
        (n0, n0_ref) -> A { (a,_), (ref a,_) => a }
        (n1, n1_ref) -> B { (_,b), (_,ref b) => b }
    }

    (Tuple3, ImmutableTuple3) {
        (n0, n0_ref) -> A { (a,_,_), (ref a,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_), (_,ref b,_) => b }
        (n2, n2_ref) -> C { (_,_,c), (_,_,ref c) => c }
    }

    (Tuple4, ImmutableTuple4) {
        (n0, n0_ref) -> A { (a,_,_,_), (ref a,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_), (_,ref b,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_), (_,_,ref c,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d), (_,_,_,ref d) => d }
    }

    (Tuple5, ImmutableTuple5) {
        (n0, n0_ref) -> A { (a,_,_,_,_), (ref a,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_), (_,ref b,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_), (_,_,ref c,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_), (_,_,_,ref d,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e), (_,_,_,_,ref e) => e }
    }

    (Tuple6, ImmutableTuple6) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_), (ref a,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_), (_,ref b,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_), (_,_,ref c,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_), (_,_,_,ref d,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_), (_,_,_,_,ref e,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f), (_,_,_,_,_,ref f) => f }
    }

    (Tuple7, ImmutableTuple7) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_), (ref a,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_), (_,ref b,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_), (_,_,ref c,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_), (_,_,_,ref d,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_), (_,_,_,_,ref e,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_), (_,_,_,_,_,ref f,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g), (_,_,_,_,_,_,ref g) => g }
    }

    (Tuple8, ImmutableTuple8) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_), (_,_,ref c,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_), (_,_,_,ref d,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_), (_,_,_,_,ref e,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_), (_,_,_,_,_,ref f,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_), (_,_,_,_,_,_,ref g,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h), (_,_,_,_,_,_,_,ref h) => h }
    }

    (Tuple9, ImmutableTuple9) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_,_), (_,_,_,_,ref e,_,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_,_), (_,_,_,_,_,ref f,_,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_,_), (_,_,_,_,_,_,ref g,_,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h,_), (_,_,_,_,_,_,_,ref h,_) => h }
        (n8, n8_ref) -> I { (_,_,_,_,_,_,_,_,i), (_,_,_,_,_,_,_,_,ref i) => i }
    }

    (Tuple10, ImmutableTuple10) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_,_,_), (_,_,_,_,_,_,ref g,_,_,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h,_,_), (_,_,_,_,_,_,_,ref h,_,_) => h }
        (n8, n8_ref) -> I { (_,_,_,_,_,_,_,_,i,_), (_,_,_,_,_,_,_,_,ref i,_) => i }
        (n9, n9_ref) -> J { (_,_,_,_,_,_,_,_,_,j), (_,_,_,_,_,_,_,_,_,ref j) => j }
    }

    (Tuple11, ImmutableTuple11) {
        (n0,  n0_ref)  -> A { (a,_,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_,_) => a }
        (n1,  n1_ref)  -> B { (_,b,_,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_,_) => b }
        (n2,  n2_ref)  -> C { (_,_,c,_,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_,_) => c }
        (n3,  n3_ref)  -> D { (_,_,_,d,_,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_,_) => d }
        (n4,  n4_ref)  -> E { (_,_,_,_,e,_,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_,_) => e }
        (n5,  n5_ref)  -> F { (_,_,_,_,_,f,_,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_,_) => f }
        (n6,  n6_ref)  -> G { (_,_,_,_,_,_,g,_,_,_,_), (_,_,_,_,_,_,ref g,_,_,_,_) => g }
        (n7,  n7_ref)  -> H { (_,_,_,_,_,_,_,h,_,_,_), (_,_,_,_,_,_,_,ref h,_,_,_) => h }
        (n8,  n8_ref)  -> I { (_,_,_,_,_,_,_,_,i,_,_), (_,_,_,_,_,_,_,_,ref i,_,_) => i }
        (n9,  n9_ref)  -> J { (_,_,_,_,_,_,_,_,_,j,_), (_,_,_,_,_,_,_,_,_,ref j,_) => j }
        (n10, n10_ref) -> K { (_,_,_,_,_,_,_,_,_,_,k), (_,_,_,_,_,_,_,_,_,_,ref k) => k }
    }

    (Tuple12, ImmutableTuple12) {
        (n0,  n0_ref)  -> A { (a,_,_,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_,_,_) => a }
        (n1,  n1_ref)  -> B { (_,b,_,_,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_,_,_) => b }
        (n2,  n2_ref)  -> C { (_,_,c,_,_,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_,_,_) => c }
        (n3,  n3_ref)  -> D { (_,_,_,d,_,_,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_,_,_) => d }
        (n4,  n4_ref)  -> E { (_,_,_,_,e,_,_,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_,_,_) => e }
        (n5,  n5_ref)  -> F { (_,_,_,_,_,f,_,_,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_,_,_) => f }
        (n6,  n6_ref)  -> G { (_,_,_,_,_,_,g,_,_,_,_,_), (_,_,_,_,_,_,ref g,_,_,_,_,_) => g }
        (n7,  n7_ref)  -> H { (_,_,_,_,_,_,_,h,_,_,_,_), (_,_,_,_,_,_,_,ref h,_,_,_,_) => h }
        (n8,  n8_ref)  -> I { (_,_,_,_,_,_,_,_,i,_,_,_), (_,_,_,_,_,_,_,_,ref i,_,_,_) => i }
        (n9,  n9_ref)  -> J { (_,_,_,_,_,_,_,_,_,j,_,_), (_,_,_,_,_,_,_,_,_,ref j,_,_) => j }
        (n10, n10_ref) -> K { (_,_,_,_,_,_,_,_,_,_,k,_), (_,_,_,_,_,_,_,_,_,_,ref k,_) => k }
        (n11, n11_ref) -> L { (_,_,_,_,_,_,_,_,_,_,_,l), (_,_,_,_,_,_,_,_,_,_,_,ref l) => l }
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
    fn test_n_tuple() {
        let t = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
        assert_eq!(t.n0(), 0u8);
        assert_eq!(t.n1(), 1u16);
        assert_eq!(t.n2(), 2u32);
        assert_eq!(t.n3(), 3u64);
        assert_eq!(t.n4(), 4u);
        assert_eq!(t.n5(), 5i8);
        assert_eq!(t.n6(), 6i16);
        assert_eq!(t.n7(), 7i32);
        assert_eq!(t.n8(), 8i64);
        assert_eq!(t.n9(), 9i);
        assert_eq!(t.n10(), 10f32);
        assert_eq!(t.n11(), 11f64);

        assert_eq!(t.n0_ref(), &0u8);
        assert_eq!(t.n1_ref(), &1u16);
        assert_eq!(t.n2_ref(), &2u32);
        assert_eq!(t.n3_ref(), &3u64);
        assert_eq!(t.n4_ref(), &4u);
        assert_eq!(t.n5_ref(), &5i8);
        assert_eq!(t.n6_ref(), &6i16);
        assert_eq!(t.n7_ref(), &7i32);
        assert_eq!(t.n8_ref(), &8i64);
        assert_eq!(t.n9_ref(), &9i);
        assert_eq!(t.n10_ref(), &10f32);
        assert_eq!(t.n11_ref(), &11f64);
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
