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

use kinds::Copy;
use vec;

pub use self::inner::*;

pub trait CopyableTuple<T, U> {
    fn first(&self) -> T;
    fn second(&self) -> U;
    fn swap(&self) -> (U, T);
}

impl<T:Copy,U:Copy> CopyableTuple<T, U> for (T, U) {

    /// Return the first element of self
    #[inline(always)]
    fn first(&self) -> T {
        let (t, _) = *self;
        return t;
    }

    /// Return the second element of self
    #[inline(always)]
    fn second(&self) -> U {
        let (_, u) = *self;
        return u;
    }

    /// Return the results of swapping the two elements of self
    #[inline(always)]
    fn swap(&self) -> (U, T) {
        let (t, u) = *self;
        return (u, t);
    }
}

pub trait ImmutableTuple<T, U> {
    fn first_ref<'a>(&'a self) -> &'a T;
    fn second_ref<'a>(&'a self) -> &'a U;
}

impl<T, U> ImmutableTuple<T, U> for (T, U) {
    #[inline(always)]
    fn first_ref<'a>(&'a self) -> &'a T {
        match *self {
            (ref t, _) => t,
        }
    }
    #[inline(always)]
    fn second_ref<'a>(&'a self) -> &'a U {
        match *self {
            (_, ref u) => u,
        }
    }
}

pub trait ExtendedTupleOps<A,B> {
    fn zip(&self) -> ~[(A, B)];
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C];
}

impl<'self,A:Copy,B:Copy> ExtendedTupleOps<A,B> for (&'self [A], &'self [B]) {
    #[inline(always)]
    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    #[inline(always)]
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map_zip(*a, *b, f)
            }
        }
    }
}

impl<A:Copy,B:Copy> ExtendedTupleOps<A,B> for (~[A], ~[B]) {

    #[inline(always)]
    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    #[inline(always)]
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map_zip(*a, *b, f)
            }
        }
    }
}

// macro for implementing n-ary tuple functions and operations

macro_rules! tuple_impls(
    ($(
        $name:ident {
            $(fn $get_fn:ident -> $T:ident { $get_pattern:pat => $ret:expr })+
        }
    )+) => (
        pub mod inner {
            use clone::Clone;
            #[cfg(not(test))] use cmp::{Eq, Ord};

            $(
                pub trait $name<$($T),+> {
                    $(fn $get_fn<'a>(&'a self) -> &'a $T;)+
                }

                impl<$($T),+> $name<$($T),+> for ($($T),+) {
                    $(
                        #[inline(always)]
                        fn $get_fn<'a>(&'a self) -> &'a $T {
                            match *self {
                                $get_pattern => $ret
                            }
                        }
                    )+
                }

                impl<$($T:Clone),+> Clone for ($($T),+) {
                    fn clone(&self) -> ($($T),+) {
                        ($(self.$get_fn().clone()),+)
                    }
                }

                #[cfg(not(test))]
                impl<$($T:Eq),+> Eq for ($($T),+) {
                    #[inline(always)]
                    fn eq(&self, other: &($($T),+)) -> bool {
                        $(*self.$get_fn() == *other.$get_fn())&&+
                    }

                    #[inline(always)]
                    fn ne(&self, other: &($($T),+)) -> bool {
                        !(*self == *other)
                    }
                }

                #[cfg(not(test))]
                impl<$($T:Ord),+> Ord for ($($T),+) {
                    #[inline(always)]
                    fn lt(&self, other: &($($T),+)) -> bool {
                        $(*self.$get_fn() < *other.$get_fn())&&+
                    }

                    #[inline(always)]
                    fn le(&self, other: &($($T),+)) -> bool {
                        $(*self.$get_fn() <= *other.$get_fn())&&+
                    }

                    #[inline(always)]
                    fn ge(&self, other: &($($T),+)) -> bool {
                        $(*self.$get_fn() >= *other.$get_fn())&&+
                    }

                    #[inline(always)]
                    fn gt(&self, other: &($($T),+)) -> bool {
                        $(*self.$get_fn() > *other.$get_fn())&&+
                    }
                }
            )+
        }
    )
)

tuple_impls!(
    Tuple2 {
        fn n0 -> A { (ref a,_) => a }
        fn n1 -> B { (_,ref b) => b }
    }

    Tuple3 {
        fn n0 -> A { (ref a,_,_) => a }
        fn n1 -> B { (_,ref b,_) => b }
        fn n2 -> C { (_,_,ref c) => c }
    }

    Tuple4 {
        fn n0 -> A { (ref a,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_) => b }
        fn n2 -> C { (_,_,ref c,_) => c }
        fn n3 -> D { (_,_,_,ref d) => d }
    }

    Tuple5 {
        fn n0 -> A { (ref a,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_) => d }
        fn n4 -> E { (_,_,_,_,ref e) => e }
    }

    Tuple6 {
        fn n0 -> A { (ref a,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f) => f }
    }

    Tuple7 {
        fn n0 -> A { (ref a,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g) => g }
    }

    Tuple8 {
        fn n0 -> A { (ref a,_,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g,_) => g }
        fn n7 -> H { (_,_,_,_,_,_,_,ref h) => h }
    }

    Tuple9 {
        fn n0 -> A { (ref a,_,_,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_,_,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g,_,_) => g }
        fn n7 -> H { (_,_,_,_,_,_,_,ref h,_) => h }
        fn n8 -> I { (_,_,_,_,_,_,_,_,ref i) => i }
    }

    Tuple10 {
        fn n0 -> A { (ref a,_,_,_,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_,_,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_,_,_,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g,_,_,_) => g }
        fn n7 -> H { (_,_,_,_,_,_,_,ref h,_,_) => h }
        fn n8 -> I { (_,_,_,_,_,_,_,_,ref i,_) => i }
        fn n9 -> J { (_,_,_,_,_,_,_,_,_,ref j) => j }
    }

    Tuple11 {
        fn n0 -> A { (ref a,_,_,_,_,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_,_,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_,_,_,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_,_,_,_,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g,_,_,_,_) => g }
        fn n7 -> H { (_,_,_,_,_,_,_,ref h,_,_,_) => h }
        fn n8 -> I { (_,_,_,_,_,_,_,_,ref i,_,_) => i }
        fn n9 -> J { (_,_,_,_,_,_,_,_,_,ref j,_) => j }
        fn n10 -> K { (_,_,_,_,_,_,_,_,_,_,ref k) => k }
    }

    Tuple12 {
        fn n0 -> A { (ref a,_,_,_,_,_,_,_,_,_,_,_) => a }
        fn n1 -> B { (_,ref b,_,_,_,_,_,_,_,_,_,_) => b }
        fn n2 -> C { (_,_,ref c,_,_,_,_,_,_,_,_,_) => c }
        fn n3 -> D { (_,_,_,ref d,_,_,_,_,_,_,_,_) => d }
        fn n4 -> E { (_,_,_,_,ref e,_,_,_,_,_,_,_) => e }
        fn n5 -> F { (_,_,_,_,_,ref f,_,_,_,_,_,_) => f }
        fn n6 -> G { (_,_,_,_,_,_,ref g,_,_,_,_,_) => g }
        fn n7 -> H { (_,_,_,_,_,_,_,ref h,_,_,_,_) => h }
        fn n8 -> I { (_,_,_,_,_,_,_,_,ref i,_,_,_) => i }
        fn n9 -> J { (_,_,_,_,_,_,_,_,_,ref j,_,_) => j }
        fn n10 -> K { (_,_,_,_,_,_,_,_,_,_,ref k,_) => k }
        fn n11 -> L { (_,_,_,_,_,_,_,_,_,_,_,ref l) => l }
    }
)

#[test]
fn test_tuple_ref() {
    let x = (~"foo", ~"bar");
    assert_eq!(x.first_ref(), &~"foo");
    assert_eq!(x.second_ref(), &~"bar");
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
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
    assert_eq!(*t.n0(), 0u8);
    assert_eq!(*t.n1(), 1u16);
    assert_eq!(*t.n2(), 2u32);
    assert_eq!(*t.n3(), 3u64);
    assert_eq!(*t.n4(), 4u);
    assert_eq!(*t.n5(), 5i8);
    assert_eq!(*t.n6(), 6i16);
    assert_eq!(*t.n7(), 7i32);
    assert_eq!(*t.n8(), 8i64);
    assert_eq!(*t.n9(), 9i);
    assert_eq!(*t.n10(), 10f32);
    assert_eq!(*t.n11(), 11f64);
}
