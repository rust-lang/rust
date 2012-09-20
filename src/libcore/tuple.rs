// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Operations on tuples

use cmp::{Eq, Ord};

trait TupleOps<T,U> {
    pure fn first() -> T;
    pure fn second() -> U;
    pure fn swap() -> (U, T);
}

impl<T: Copy, U: Copy> (T, U): TupleOps<T,U> {

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

trait ExtendedTupleOps<A,B> {
    fn zip() -> ~[(A, B)];
    fn map<C>(f: fn(A, B) -> C) -> ~[C];
}

impl<A: Copy, B: Copy> (&[A], &[B]): ExtendedTupleOps<A,B> {

    fn zip() -> ~[(A, B)] {
        let (a, b) = self;
        vec::zip_slice(a, b)
    }

    fn map<C>(f: fn(A, B) -> C) -> ~[C] {
        let (a, b) = self;
        vec::map2(a, b, f)
    }
}

impl<A: Copy, B: Copy> (~[A], ~[B]): ExtendedTupleOps<A,B> {

    fn zip() -> ~[(A, B)] {
        // FIXME #2543: Bad copy
        let (a, b) = copy self;
        vec::zip(move a, move b)
    }

    fn map<C>(f: fn(A, B) -> C) -> ~[C] {
        // FIXME #2543: Bad copy
        let (a, b) = copy self;
        vec::map2(a, b, f)
    }
}

#[cfg(stage0)]
impl<A: Eq, B: Eq> (A, B): Eq {
    pure fn eq(&&other: (A, B)) -> bool {
        // XXX: This would be a lot less wordy with ref bindings, but I don't
        // trust that they work yet.
        match self {
            (self_a, self_b) => {
                match other {
                    (other_a, other_b) => {
                        self_a.eq(other_a) && self_b.eq(other_b)
                    }
                }
            }
        }
    }
    pure fn ne(&&other: (A, B)) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<A: Eq, B: Eq> (A, B) : Eq {
    pure fn eq(other: &(A, B)) -> bool {
        // XXX: This would be a lot less wordy with ref bindings, but I don't
        // trust that they work yet.
        match self {
            (self_a, self_b) => {
                match (*other) {
                    (ref other_a, ref other_b) => {
                        self_a.eq(other_a) && self_b.eq(other_b)
                    }
                }
            }
        }
    }
    pure fn ne(other: &(A, B)) -> bool { !self.eq(other) }
}

#[cfg(stage0)]
impl<A: Ord, B: Ord> (A, B): Ord {
    pure fn lt(&&other: (A, B)) -> bool {
        match self {
            (self_a, self_b) => {
                match other {
                    (other_a, other_b) => {
                        if self_a.lt(other_a) { return true; }
                        if other_a.lt(self_a) { return false; }
                        if self_b.lt(other_b) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    pure fn le(&&other: (A, B)) -> bool { !other.lt(self) }
    pure fn ge(&&other: (A, B)) -> bool { !self.lt(other) }
    pure fn gt(&&other: (A, B)) -> bool { other.lt(self)  }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<A: Ord, B: Ord> (A, B) : Ord {
    pure fn lt(other: &(A, B)) -> bool {
        match self {
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
    pure fn le(other: &(A, B)) -> bool { !(*other).lt(&self) }
    pure fn ge(other: &(A, B)) -> bool { !self.lt(other) }
    pure fn gt(other: &(A, B)) -> bool { (*other).lt(&self)  }
}

#[cfg(stage0)]
impl<A: Eq, B: Eq, C: Eq> (A, B, C): Eq {
    pure fn eq(&&other: (A, B, C)) -> bool {
        // XXX: This would be a lot less wordy with ref bindings, but I don't
        // trust that they work yet.
        match self {
            (self_a, self_b, self_c) => {
                match other {
                    (other_a, other_b, other_c) => {
                        self_a.eq(other_a) &&
                        self_b.eq(other_b) &&
                        self_c.eq(other_c)
                    }
                }
            }
        }
    }
    pure fn ne(&&other: (A, B, C)) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<A: Eq, B: Eq, C: Eq> (A, B, C) : Eq {
    pure fn eq(other: &(A, B, C)) -> bool {
        // XXX: This would be a lot less wordy with ref bindings, but I don't
        // trust that they work yet.
        match self {
            (self_a, self_b, self_c) => {
                match (*other) {
                    (ref other_a, ref other_b, ref other_c) => {
                        self_a.eq(other_a) &&
                        self_b.eq(other_b) &&
                        self_c.eq(other_c)
                    }
                }
            }
        }
    }
    pure fn ne(other: &(A, B, C)) -> bool { !self.eq(other) }
}

#[cfg(stage0)]
impl<A: Ord, B: Ord, C: Ord> (A, B, C): Ord {
    pure fn lt(&&other: (A, B, C)) -> bool {
        match self {
            (self_a, self_b, self_c) => {
                match other {
                    (other_a, other_b, other_c) => {
                        if self_a.lt(other_a) { return true; }
                        if other_a.lt(self_a) { return false; }
                        if self_b.lt(other_b) { return true; }
                        if other_b.lt(self_b) { return false; }
                        if self_c.lt(other_c) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    pure fn le(&&other: (A, B, C)) -> bool { !other.lt(self) }
    pure fn ge(&&other: (A, B, C)) -> bool { !self.lt(other) }
    pure fn gt(&&other: (A, B, C)) -> bool { other.lt(self)  }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<A: Ord, B: Ord, C: Ord> (A, B, C) : Ord {
    pure fn lt(other: &(A, B, C)) -> bool {
        match self {
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
    pure fn le(other: &(A, B, C)) -> bool { !(*other).lt(&self) }
    pure fn ge(other: &(A, B, C)) -> bool { !self.lt(other) }
    pure fn gt(other: &(A, B, C)) -> bool { (*other).lt(&self)  }
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
fn test_tuple() {
    assert (948, 4039.48).first() == 948;
    assert (34.5, ~"foo").second() == ~"foo";
    assert ('a', 2).swap() == (2, 'a');
}

