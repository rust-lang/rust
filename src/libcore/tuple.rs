// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Operations on tuples

use cmp::{Eq, Ord};

pub trait CopyableTuple<T, U> {
    pure fn first() -> T;
    pure fn second() -> U;
    pure fn swap() -> (U, T);
}

impl<T: Copy, U: Copy> (T, U): CopyableTuple<T, U> {

    /// Return the first element of self
    pure fn first() -> T {
        let (t, _) = self;
        return t;
    }

    /// Return the second element of self
    pure fn second() -> U {
        let (_, u) = self;
        return u;
    }

    /// Return the results of swapping the two elements of self
    pure fn swap() -> (U, T) {
        let (t, u) = self;
        return (u, t);
    }

}

pub trait ImmutableTuple<T, U> {
    pure fn first_ref(&self) -> &self/T;
    pure fn second_ref(&self) -> &self/U;
}

impl<T, U> (T, U): ImmutableTuple<T, U> {
    pure fn first_ref(&self) -> &self/T {
        match *self {
            (ref t, _) => t,
        }
    }
    pure fn second_ref(&self) -> &self/U {
        match *self {
            (_, ref u) => u,
        }
    }
}

pub trait ExtendedTupleOps<A,B> {
    fn zip(&self) -> ~[(A, B)];
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C];
}

impl<A: Copy, B: Copy> (&[A], &[B]): ExtendedTupleOps<A,B> {
    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map2(*a, *b, f)
            }
        }
    }
}

impl<A: Copy, B: Copy> (~[A], ~[B]): ExtendedTupleOps<A,B> {

    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map2(*a, *b, f)
            }
        }
    }
}

impl<A: Eq, B: Eq> (A, B) : Eq {
    pure fn eq(&self, other: &(A, B)) -> bool {
        match (*self) {
            (ref self_a, ref self_b) => match other {
                &(ref other_a, ref other_b) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                }
            }
        }
    }
    pure fn ne(&self, other: &(A, B)) -> bool { !(*self).eq(other) }
}

impl<A: Ord, B: Ord> (A, B) : Ord {
    pure fn lt(&self, other: &(A, B)) -> bool {
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
    pure fn le(&self, other: &(A, B)) -> bool { !(*other).lt(&(*self)) }
    pure fn ge(&self, other: &(A, B)) -> bool { !(*self).lt(other) }
    pure fn gt(&self, other: &(A, B)) -> bool { (*other).lt(&(*self))  }
}

impl<A: Eq, B: Eq, C: Eq> (A, B, C) : Eq {
    pure fn eq(&self, other: &(A, B, C)) -> bool {
        match (*self) {
            (ref self_a, ref self_b, ref self_c) => match other {
                &(ref other_a, ref other_b, ref other_c) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                        && (*self_c).eq(other_c)
                }
            }
        }
    }
    pure fn ne(&self, other: &(A, B, C)) -> bool { !(*self).eq(other) }
}

impl<A: Ord, B: Ord, C: Ord> (A, B, C) : Ord {
    pure fn lt(&self, other: &(A, B, C)) -> bool {
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
    pure fn le(&self, other: &(A, B, C)) -> bool { !(*other).lt(&(*self)) }
    pure fn ge(&self, other: &(A, B, C)) -> bool { !(*self).lt(other) }
    pure fn gt(&self, other: &(A, B, C)) -> bool { (*other).lt(&(*self))  }
}

#[test]
fn test_tuple_ref() {
    let x = (~"foo", ~"bar");
    assert x.first_ref() == &~"foo";
    assert x.second_ref() == &~"bar";
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
fn test_tuple() {
    assert (948, 4039.48).first() == 948;
    assert (34.5, ~"foo").second() == ~"foo";
    assert ('a', 2).swap() == (2, 'a');
}

