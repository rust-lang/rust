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

n_tuple!(Tuple2: n0:A, n1:B)
n_tuple!(Tuple3: n0:A, n1:B, n2:C)
n_tuple!(Tuple4: n0:A, n1:B, n2:C, n3:D)
n_tuple!(Tuple5: n0:A, n1:B, n2:C, n3:D, n4:E)
n_tuple!(Tuple6: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F)
n_tuple!(Tuple7: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G)
n_tuple!(Tuple8: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G, n7:H)
n_tuple!(Tuple9: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G, n7:H, n8:I)
n_tuple!(Tuple10: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G, n7:H, n8:I, n9:J)
n_tuple!(Tuple11: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G, n7:H, n8:I, n9:J, n10:K)
n_tuple!(Tuple12: n0:A, n1:B, n2:C, n3:D, n4:E, n5:F, n6:G, n7:H, n8:I, n9:J, n10:K, n11:L)

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
    n0 -> A { (a,_) => a }
    n1 -> B { (_,b) => b }
)

impl_n_tuple!(Tuple3:
    n0 -> A { (a,_,_) => a }
    n1 -> B { (_,b,_) => b }
    n2 -> C { (_,_,c) => c }
)

impl_n_tuple!(Tuple4:
    n0 -> A { (a,_,_,_) => a }
    n1 -> B { (_,b,_,_) => b }
    n2 -> C { (_,_,c,_) => c }
    n3 -> D { (_,_,_,d) => d }
)

impl_n_tuple!(Tuple5:
    n0 -> A { (a,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_) => b }
    n2 -> C { (_,_,c,_,_) => c }
    n3 -> D { (_,_,_,d,_) => d }
    n4 -> E { (_,_,_,_,e) => e }
)

impl_n_tuple!(Tuple6:
    n0 -> A { (a,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_) => d }
    n4 -> E { (_,_,_,_,e,_) => e }
    n5 -> F { (_,_,_,_,_,f) => f }
)

impl_n_tuple!(Tuple7:
    n0 -> A { (a,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_) => f }
    n6 -> G { (_,_,_,_,_,_,g) => g }
)

impl_n_tuple!(Tuple8:
    n0 -> A { (a,_,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_,_) => f }
    n6 -> G { (_,_,_,_,_,_,g,_) => g }
    n7 -> H { (_,_,_,_,_,_,_,h) => h }
)

impl_n_tuple!(Tuple9:
    n0 -> A { (a,_,_,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_,_,_) => f }
    n6 -> G { (_,_,_,_,_,_,g,_,_) => g }
    n7 -> H { (_,_,_,_,_,_,_,h,_) => h }
    n8 -> I { (_,_,_,_,_,_,_,_,i) => i }
)

impl_n_tuple!(Tuple10:
    n0 -> A { (a,_,_,_,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_,_,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_,_,_,_) => f }
    n6 -> G { (_,_,_,_,_,_,g,_,_,_) => g }
    n7 -> H { (_,_,_,_,_,_,_,h,_,_) => h }
    n8 -> I { (_,_,_,_,_,_,_,_,i,_) => i }
    n9 -> J { (_,_,_,_,_,_,_,_,_,j) => j }
)

impl_n_tuple!(Tuple11:
    n0 -> A { (a,_,_,_,_,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_,_,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_,_,_,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_,_,_,_,_) => f }
    n6 -> G { (_,_,_,_,_,_,g,_,_,_,_) => g }
    n7 -> H { (_,_,_,_,_,_,_,h,_,_,_) => h }
    n8 -> I { (_,_,_,_,_,_,_,_,i,_,_) => i }
    n9 -> J { (_,_,_,_,_,_,_,_,_,j,_) => j }
    n10 -> K { (_,_,_,_,_,_,_,_,_,_,k) => k }
)

impl_n_tuple!(Tuple12:
    n0 -> A { (a,_,_,_,_,_,_,_,_,_,_,_) => a }
    n1 -> B { (_,b,_,_,_,_,_,_,_,_,_,_) => b }
    n2 -> C { (_,_,c,_,_,_,_,_,_,_,_,_) => c }
    n3 -> D { (_,_,_,d,_,_,_,_,_,_,_,_) => d }
    n4 -> E { (_,_,_,_,e,_,_,_,_,_,_,_) => e }
    n5 -> F { (_,_,_,_,_,f,_,_,_,_,_,_) => f }
    n6 -> G { (_,_,_,_,_,_,g,_,_,_,_,_) => g }
    n7 -> H { (_,_,_,_,_,_,_,h,_,_,_,_) => h }
    n8 -> I { (_,_,_,_,_,_,_,_,i,_,_,_) => i }
    n9 -> J { (_,_,_,_,_,_,_,_,_,j,_,_) => j }
    n10 -> K { (_,_,_,_,_,_,_,_,_,_,k,_) => k }
    n11 -> L { (_,_,_,_,_,_,_,_,_,_,_,l) => l }
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
}
