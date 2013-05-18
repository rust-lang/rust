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

use clone::Clone;
use kinds::Copy;
use vec;

#[cfg(not(test))] use cmp::{Eq, Ord};

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

impl<T:Clone,U:Clone> Clone for (T, U) {
    fn clone(&self) -> (T, U) {
        let (a, b) = match *self {
            (ref a, ref b) => (a, b)
        };
        (a.clone(), b.clone())
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

#[cfg(not(test))]
impl<A:Eq> Eq for (A,) {
    #[inline(always)]
    fn eq(&self, other: &(A,)) -> bool {
        match (*self) {
            (ref self_a,) => match other {
                &(ref other_a,) => {
                    (*self_a).eq(other_a)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A,)) -> bool { !(*self).eq(other) }
}

#[cfg(not(test))]
impl<A:Ord> Ord for (A,) {
    #[inline(always)]
    fn lt(&self, other: &(A,)) -> bool {
        match (*self) {
            (ref self_a,) => {
                match (*other) {
                    (ref other_a,) => {
                        if (*self_a).lt(other_a) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A,)) -> bool { !other.lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A,)) -> bool { !self.lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A,)) -> bool { other.lt(&(*self))  }
}

#[cfg(not(test))]
impl<A:Eq,B:Eq> Eq for (A, B) {
    #[inline(always)]
    fn eq(&self, other: &(A, B)) -> bool {
        match (*self) {
            (ref self_a, ref self_b) => match other {
                &(ref other_a, ref other_b) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A, B)) -> bool { !(*self).eq(other) }
}

#[cfg(not(test))]
impl<A:Ord,B:Ord> Ord for (A, B) {
    #[inline(always)]
    fn lt(&self, other: &(A, B)) -> bool {
        match (*self) {
            (ref self_a, ref self_b) => {
                match (*other) {
                    (ref other_a, ref other_b) => {
                        if (*self_a).lt(other_a) { return true; }
                        if (*other_a).lt(self_a) { return false; }
                        if (*self_b).lt(other_b) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A, B)) -> bool { !(*other).lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A, B)) -> bool { !(*self).lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A, B)) -> bool { (*other).lt(&(*self))  }
}

#[cfg(not(test))]
impl<A:Eq,B:Eq,C:Eq> Eq for (A, B, C) {
    #[inline(always)]
    fn eq(&self, other: &(A, B, C)) -> bool {
        match (*self) {
            (ref self_a, ref self_b, ref self_c) => match other {
                &(ref other_a, ref other_b, ref other_c) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                        && (*self_c).eq(other_c)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A, B, C)) -> bool { !(*self).eq(other) }
}

#[cfg(not(test))]
impl<A:Ord,B:Ord,C:Ord> Ord for (A, B, C) {
    #[inline(always)]
    fn lt(&self, other: &(A, B, C)) -> bool {
        match (*self) {
            (ref self_a, ref self_b, ref self_c) => {
                match (*other) {
                    (ref other_a, ref other_b, ref other_c) => {
                        if (*self_a).lt(other_a) { return true; }
                        if (*other_a).lt(self_a) { return false; }
                        if (*self_b).lt(other_b) { return true; }
                        if (*other_b).lt(self_b) { return false; }
                        if (*self_c).lt(other_c) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A, B, C)) -> bool { !(*other).lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A, B, C)) -> bool { !(*self).lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A, B, C)) -> bool { (*other).lt(&(*self))  }
}

// Tuple element accessor traits

macro_rules! n_tuple(
    ($name:ident: $($method:ident : $T:ident),+) => (
        pub trait $name<$($T),+> {
            $(fn $method(&self) -> $T;)+
        }
    )
)

n_tuple!(Tuple2: _0:A, _1:B)
n_tuple!(Tuple3: _0:A, _1:B, _2:C)
n_tuple!(Tuple4: _0:A, _1:B, _2:C, _3:D)
n_tuple!(Tuple5: _0:A, _1:B, _2:C, _3:D, _4:E)
n_tuple!(Tuple6: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F)
n_tuple!(Tuple7: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G)
n_tuple!(Tuple8: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G, _7:H)
n_tuple!(Tuple9: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G, _7:H, _8:I)
n_tuple!(Tuple10: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G, _7:H, _8:I, _9:J)
n_tuple!(Tuple11: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G, _7:H, _8:I, _9:J, _10:K)
n_tuple!(Tuple12: _0:A, _1:B, _2:C, _3:D, _4:E, _5:F, _6:G, _7:H, _8:I, _9:J, _10:K, _11:L)

// Tuple element accessor trait implementations

macro_rules! impl_n_tuple(
    ($name:ident: $($method:ident -> $T:ident { $accessor:pat => $t:expr })+) => (
        impl<$($T:Copy),+> $name<$($T),+> for ($($T),+) {
            $(
                fn $method(&self) -> $T {
                    match *self {
                        $accessor => $t
                    }
                }
            )+
        }
    )
)

impl_n_tuple!(Tuple2:
    _0 -> A { (a,_) => a }
    _1 -> B { (_,b) => b }
)

impl_n_tuple!(Tuple3:
    _0 -> A { (a,_,_) => a }
    _1 -> B { (_,b,_) => b }
    _2 -> C { (_,_,c) => c }
)

impl_n_tuple!(Tuple4:
    _0 -> A { (a,_,_,_) => a }
    _1 -> B { (_,b,_,_) => b }
    _2 -> C { (_,_,c,_) => c }
    _3 -> D { (_,_,_,d) => d }
)

impl_n_tuple!(Tuple5:
    _0 -> A { (a,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_) => b }
    _2 -> C { (_,_,c,_,_) => c }
    _3 -> D { (_,_,_,d,_) => d }
    _4 -> E { (_,_,_,_,e) => e }
)

impl_n_tuple!(Tuple6:
    _0 -> A { (a,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_) => d }
    _4 -> E { (_,_,_,_,e,_) => e }
    _5 -> F { (_,_,_,_,_,f) => f }
)

impl_n_tuple!(Tuple7:
    _0 -> A { (a,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_) => f }
    _6 -> G { (_,_,_,_,_,_,g) => g }
)

impl_n_tuple!(Tuple8:
    _0 -> A { (a,_,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_,_) => f }
    _6 -> G { (_,_,_,_,_,_,g,_) => g }
    _7 -> H { (_,_,_,_,_,_,_,h) => h }
)

impl_n_tuple!(Tuple9:
    _0 -> A { (a,_,_,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_,_,_) => f }
    _6 -> G { (_,_,_,_,_,_,g,_,_) => g }
    _7 -> H { (_,_,_,_,_,_,_,h,_) => h }
    _8 -> I { (_,_,_,_,_,_,_,_,i) => i }
)

impl_n_tuple!(Tuple10:
    _0 -> A { (a,_,_,_,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_,_,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_,_,_,_) => f }
    _6 -> G { (_,_,_,_,_,_,g,_,_,_) => g }
    _7 -> H { (_,_,_,_,_,_,_,h,_,_) => h }
    _8 -> I { (_,_,_,_,_,_,_,_,i,_) => i }
    _9 -> J { (_,_,_,_,_,_,_,_,_,j) => j }
)

impl_n_tuple!(Tuple11:
    _0 -> A { (a,_,_,_,_,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_,_,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_,_,_,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_,_,_,_,_) => f }
    _6 -> G { (_,_,_,_,_,_,g,_,_,_,_) => g }
    _7 -> H { (_,_,_,_,_,_,_,h,_,_,_) => h }
    _8 -> I { (_,_,_,_,_,_,_,_,i,_,_) => i }
    _9 -> J { (_,_,_,_,_,_,_,_,_,j,_) => j }
    _10 -> K { (_,_,_,_,_,_,_,_,_,_,k) => k }
)

impl_n_tuple!(Tuple12:
    _0 -> A { (a,_,_,_,_,_,_,_,_,_,_,_) => a }
    _1 -> B { (_,b,_,_,_,_,_,_,_,_,_,_) => b }
    _2 -> C { (_,_,c,_,_,_,_,_,_,_,_,_) => c }
    _3 -> D { (_,_,_,d,_,_,_,_,_,_,_,_) => d }
    _4 -> E { (_,_,_,_,e,_,_,_,_,_,_,_) => e }
    _5 -> F { (_,_,_,_,_,f,_,_,_,_,_,_) => f }
    _6 -> G { (_,_,_,_,_,_,g,_,_,_,_,_) => g }
    _7 -> H { (_,_,_,_,_,_,_,h,_,_,_,_) => h }
    _8 -> I { (_,_,_,_,_,_,_,_,i,_,_,_) => i }
    _9 -> J { (_,_,_,_,_,_,_,_,_,j,_,_) => j }
    _10 -> K { (_,_,_,_,_,_,_,_,_,_,k,_) => k }
    _11 -> L { (_,_,_,_,_,_,_,_,_,_,_,l) => l }
)

#[test]
fn test_tuple_ref() {
    let x = (~"foo", ~"bar");
    assert!(x.first_ref() == &~"foo");
    assert!(x.second_ref() == &~"bar");
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
fn test_tuple() {
    assert!((948, 4039.48).first() == 948);
    assert!((34.5, ~"foo").second() == ~"foo");
    assert!(('a', 2).swap() == (2, 'a'));
}

#[test]
fn test_clone() {
    let a = (1, ~"2");
    let b = a.clone();
    assert!(a.first() == b.first());
    assert!(a.second() == b.second());
}

#[test]
fn test_n_tuple() {
    let t = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
    assert_eq!(t._0(), 0u8);
    assert_eq!(t._1(), 1u16);
    assert_eq!(t._2(), 2u32);
    assert_eq!(t._3(), 3u64);
    assert_eq!(t._4(), 4u);
    assert_eq!(t._5(), 5i8);
    assert_eq!(t._6(), 6i16);
    assert_eq!(t._7(), 7i32);
    assert_eq!(t._8(), 8i64);
    assert_eq!(t._9(), 9i);
    assert_eq!(t._10(), 10f32);
    assert_eq!(t._11(), 11f64);
}
