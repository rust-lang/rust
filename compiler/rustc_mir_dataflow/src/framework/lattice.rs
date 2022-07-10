//! Traits used to represent [lattices] for use as the domain of a dataflow analysis.
//!
//! # Overview
//!
//! The most common lattice is a powerset of some set `S`, ordered by [set inclusion]. The [Hasse
//! diagram] for the powerset of a set with two elements (`X` and `Y`) is shown below. Note that
//! distinct elements at the same height in a Hasse diagram (e.g. `{X}` and `{Y}`) are
//! *incomparable*, not equal.
//!
//! ```text
//!      {X, Y}    <- top
//!       /  \
//!    {X}    {Y}
//!       \  /
//!        {}      <- bottom
//!
//! ```
//!
//! The defining characteristic of a lattice—the one that differentiates it from a [partially
//! ordered set][poset]—is the existence of a *unique* least upper and greatest lower bound for
//! every pair of elements. The lattice join operator (`∨`) returns the least upper bound, and the
//! lattice meet operator (`∧`) returns the greatest lower bound. Types that implement one operator
//! but not the other are known as semilattices. Dataflow analysis only uses the join operator and
//! will work with any join-semilattice, but both should be specified when possible.
//!
//! ## `PartialOrd`
//!
//! Given that they represent partially ordered sets, you may be surprised that [`JoinSemiLattice`]
//! and [`MeetSemiLattice`] do not have [`PartialOrd`][std::cmp::PartialOrd] as a supertrait. This
//! is because most standard library types use lexicographic ordering instead of set inclusion for
//! their `PartialOrd` impl. Since we do not actually need to compare lattice elements to run a
//! dataflow analysis, there's no need for a newtype wrapper with a custom `PartialOrd` impl. The
//! only benefit would be the ability to check that the least upper (or greatest lower) bound
//! returned by the lattice join (or meet) operator was in fact greater (or lower) than the inputs.
//!
//! [lattices]: https://en.wikipedia.org/wiki/Lattice_(order)
//! [set inclusion]: https://en.wikipedia.org/wiki/Subset
//! [Hasse diagram]: https://en.wikipedia.org/wiki/Hasse_diagram
//! [poset]: https://en.wikipedia.org/wiki/Partially_ordered_set

use crate::framework::BitSetExt;
use rustc_index::bit_set::{BitSet, ChunkedBitSet, HybridBitSet};
use rustc_index::vec::{Idx, IndexVec};
use std::iter;

/// A [partially ordered set][poset] that has a [least upper bound][lub] for any pair of elements
/// in the set.
///
/// [lub]: https://en.wikipedia.org/wiki/Infimum_and_supremum
/// [poset]: https://en.wikipedia.org/wiki/Partially_ordered_set
pub trait JoinSemiLattice: Eq {
    /// Computes the least upper bound of two elements, storing the result in `self` and returning
    /// `true` if `self` has changed.
    ///
    /// The lattice join operator is abbreviated as `∨`.
    fn join(&mut self, other: &Self) -> bool;
}

/// A [partially ordered set][poset] that has a [greatest lower bound][glb] for any pair of
/// elements in the set.
///
/// Dataflow analyses only require that their domains implement [`JoinSemiLattice`], not
/// `MeetSemiLattice`. However, types that will be used as dataflow domains should implement both
/// so that they can be used with [`Dual`].
///
/// [glb]: https://en.wikipedia.org/wiki/Infimum_and_supremum
/// [poset]: https://en.wikipedia.org/wiki/Partially_ordered_set
pub trait MeetSemiLattice: Eq {
    /// Computes the greatest lower bound of two elements, storing the result in `self` and
    /// returning `true` if `self` has changed.
    ///
    /// The lattice meet operator is abbreviated as `∧`.
    fn meet(&mut self, other: &Self) -> bool;
}

/// A `bool` is a "two-point" lattice with `true` as the top element and `false` as the bottom:
///
/// ```text
///      true
///        |
///      false
/// ```
impl JoinSemiLattice for bool {
    fn join(&mut self, other: &Self) -> bool {
        if let (false, true) = (*self, *other) {
            *self = true;
            return true;
        }

        false
    }
}

impl MeetSemiLattice for bool {
    fn meet(&mut self, other: &Self) -> bool {
        if let (true, false) = (*self, *other) {
            *self = false;
            return true;
        }

        false
    }
}

/// A tuple (or list) of lattices is itself a lattice whose least upper bound is the concatenation
/// of the least upper bounds of each element of the tuple (or list).
///
/// In other words:
///     (A₀, A₁, ..., Aₙ) ∨ (B₀, B₁, ..., Bₙ) = (A₀∨B₀, A₁∨B₁, ..., Aₙ∨Bₙ)
impl<I: Idx, T: JoinSemiLattice> JoinSemiLattice for IndexVec<I, T> {
    fn join(&mut self, other: &Self) -> bool {
        assert_eq!(self.len(), other.len());

        let mut changed = false;
        for (a, b) in iter::zip(self, other) {
            changed |= a.join(b);
        }
        changed
    }
}

impl<I: Idx, T: MeetSemiLattice> MeetSemiLattice for IndexVec<I, T> {
    fn meet(&mut self, other: &Self) -> bool {
        assert_eq!(self.len(), other.len());

        let mut changed = false;
        for (a, b) in iter::zip(self, other) {
            changed |= a.meet(b);
        }
        changed
    }
}

/// A `BitSet` represents the lattice formed by the powerset of all possible values of
/// the index type `T` ordered by inclusion. Equivalently, it is a tuple of "two-point" lattices,
/// one for each possible value of `T`.
impl<T: Idx> JoinSemiLattice for BitSet<T> {
    fn join(&mut self, other: &Self) -> bool {
        self.union(other)
    }
}

impl<T: Idx> MeetSemiLattice for BitSet<T> {
    fn meet(&mut self, other: &Self) -> bool {
        self.intersect(other)
    }
}

impl<T: Idx> JoinSemiLattice for ChunkedBitSet<T> {
    fn join(&mut self, other: &Self) -> bool {
        self.union(other)
    }
}

impl<T: Idx> MeetSemiLattice for ChunkedBitSet<T> {
    fn meet(&mut self, other: &Self) -> bool {
        self.intersect(other)
    }
}

/// The counterpart of a given semilattice `T` using the [inverse order].
///
/// The dual of a join-semilattice is a meet-semilattice and vice versa. For example, the dual of a
/// powerset has the empty set as its top element and the full set as its bottom element and uses
/// set *intersection* as its join operator.
///
/// [inverse order]: https://en.wikipedia.org/wiki/Duality_(order_theory)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dual<T>(pub T);

impl<T: Idx> BitSetExt<T> for Dual<BitSet<T>> {
    fn domain_size(&self) -> usize {
        self.0.domain_size()
    }

    fn contains(&self, elem: T) -> bool {
        self.0.contains(elem)
    }

    fn union(&mut self, other: &HybridBitSet<T>) {
        self.0.union(other);
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) {
        self.0.subtract(other);
    }
}

impl<T: MeetSemiLattice> JoinSemiLattice for Dual<T> {
    fn join(&mut self, other: &Self) -> bool {
        self.0.meet(&other.0)
    }
}

impl<T: JoinSemiLattice> MeetSemiLattice for Dual<T> {
    fn meet(&mut self, other: &Self) -> bool {
        self.0.join(&other.0)
    }
}

/// Extends a type `T` with top and bottom elements to make it a partially ordered set in which no
/// value of `T` is comparable with any other.
///
/// A flat set has the following [Hasse diagram]:
///
/// ```text
///          top
///  / ... / /  \ \ ... \
/// all possible values of `T`
///  \ ... \ \  / / ... /
///         bottom
/// ```
///
/// [Hasse diagram]: https://en.wikipedia.org/wiki/Hasse_diagram
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlatSet<T> {
    Bottom,
    Elem(T),
    Top,
}

impl<T: Clone + Eq> JoinSemiLattice for FlatSet<T> {
    fn join(&mut self, other: &Self) -> bool {
        let result = match (&*self, other) {
            (Self::Top, _) | (_, Self::Bottom) => return false,
            (Self::Elem(a), Self::Elem(b)) if a == b => return false,

            (Self::Bottom, Self::Elem(x)) => Self::Elem(x.clone()),

            _ => Self::Top,
        };

        *self = result;
        true
    }
}

impl<T: Clone + Eq> MeetSemiLattice for FlatSet<T> {
    fn meet(&mut self, other: &Self) -> bool {
        let result = match (&*self, other) {
            (Self::Bottom, _) | (_, Self::Top) => return false,
            (Self::Elem(ref a), Self::Elem(ref b)) if a == b => return false,

            (Self::Top, Self::Elem(ref x)) => Self::Elem(x.clone()),

            _ => Self::Bottom,
        };

        *self = result;
        true
    }
}

macro_rules! packed_int_join_semi_lattice {
    ($name: ident, $base: ty) => {
        #[derive(Debug, PartialEq, Eq, Copy, Clone, PartialOrd, Ord)]
        pub struct $name($base);
        impl $name {
            pub const TOP: Self = Self(<$base>::MAX);
            #[inline]
            pub const fn new(v: $base) -> Self {
                Self(v)
            }

            /// `saturating_new` will convert an arbitrary value (i.e. u32) into a Fact which
            /// may have a smaller internal representation (i.e. u8). If the value is too large,
            /// it will be converted to `TOP`, which is safe because `TOP` is the most
            /// conservative estimate, assuming no information. Note, it is _not_ safe to
            /// assume `BOT`, since this assumes information about the value.
            #[inline]
            pub fn saturating_new(v: impl TryInto<$base>) -> Self {
                v.try_into().map(|v| Self(v)).unwrap_or(Self::TOP)
            }

            pub const fn inner(self) -> $base {
                self.0
            }
        }

        impl JoinSemiLattice for $name {
            #[inline]
            fn join(&mut self, other: &Self) -> bool {
                match (*self, *other) {
                    (Self::TOP, _) => false,
                    (a, b) if a == b => false,
                    _ => {
                        *self = Self::TOP;
                        true
                    }
                }
            }
        }

        impl<C> crate::fmt::DebugWithContext<C> for $name {
            fn fmt_with(&self, _: &C, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if *self == Self::TOP { write!(f, "TOP") } else { write!(f, "{}", self.inner()) }
            }
        }
    };
}

packed_int_join_semi_lattice!(PackedU8JoinSemiLattice, u8);

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct FactArray<T, const N: usize> {
    // FIXME(julianknodt): maybe map Idxs to each N element?
    pub arr: [T; N],
}

impl<T, const N: usize> FactArray<T, N> {
    #[inline]
    pub fn insert(&mut self, i: impl Idx, fact: T) {
        let Some(v) = self.arr.get_mut(i.index()) else { return };
        *v = fact;
    }
    #[inline]
    pub fn get(&self, i: &impl Idx) -> Option<&T> {
        self.arr.get(i.index())
    }
}

impl<T: JoinSemiLattice, const N: usize> JoinSemiLattice for FactArray<T, N> {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (a, b) in self.arr.iter_mut().zip(other.arr.iter()) {
            changed |= a.join(b);
        }
        changed
    }
}

impl<T: MeetSemiLattice, const N: usize> MeetSemiLattice for FactArray<T, N> {
    fn meet(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (a, b) in self.arr.iter_mut().zip(other.arr.iter()) {
            changed |= a.meet(b);
        }
        changed
    }
}

/// FactCache is a struct that contains `N` recent facts (of type F) from dataflow analysis,
/// where a fact is information about some component of a program, such as the possible values a
/// variable can take. Variables are indexed by `I: Idx` (i.e. mir::Local), and `L` represents
/// location/recency, so that when merging two fact caches, the more recent information takes
/// precedence.
/// This representation is used because it takes constant memory, and assumes that recent facts
/// will have temporal locality (i.e. will be used closed to where they are generated). Thus, it
/// is more conservative than a complete analysis, but should be fast.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct FactCache<I, L, F, const N: usize> {
    facts: [F; N],
    ord: [(I, L); N],
    len: usize,
}

impl<I: Idx, L: Ord + Eq + Copy, F, const N: usize> FactCache<I, L, F, N> {
    pub fn new(empty_i: I, empty_l: L, empty_f: F) -> Self
    where
        F: Copy,
    {
        Self { facts: [empty_f; N], ord: [(empty_i, empty_l); N], len: 0 }
    }
    /// (nserts a fact into the cache, evicting the oldest one,
    /// Or updating it if there is information on one already. If the new fact being
    /// inserted is older than the previous fact, it will not be inserted.
    pub fn insert(&mut self, i: I, l: L, fact: F) {
        let mut idx = None;
        for (j, (ci, _cl)) in self.ord[..self.len].iter_mut().enumerate() {
            if *ci == i {
                // if an older fact is inserted, still update the cache: i.e. cl <= l usually
                // but this is broken during apply switch int edge effects, because the engine
                // may choose an arbitrary order for basic blocks to apply it to.
                idx = Some(j);
                break;
            }
        }
        if idx.is_none() && self.len < N {
            let new_len = self.len + 1;
            idx = Some(std::mem::replace(&mut self.len, new_len));
        };
        if let Some(idx) = idx {
            self.facts[idx] = fact;
            self.ord[idx] = (i, l);
            return;
        };
        let (p, (_, old_l)) = self.ord.iter().enumerate().min_by_key(|k| k.1.1).unwrap();
        // FIXME(julianknodt) maybe don't make this an assert but just don't update?
        assert!(*old_l <= l);
        self.ord[p] = (i, l);
        self.facts[p] = fact;
    }
    pub fn get(&self, i: I) -> Option<(&L, &F)> {
        let (p, (_, loc)) =
            self.ord[..self.len].iter().enumerate().find(|(_, iloc)| iloc.0 == i)?;
        Some((loc, &self.facts[p]))
    }
    pub fn remove(&mut self, i: I) -> bool {
        let Some(pos) = self.ord[..self.len].iter().position(|(ci, _)| *ci == i)
        else { return false };

        self.remove_idx(pos);
        return true;
    }
    #[inline]
    fn remove_idx(&mut self, i: usize) {
        assert!(i < self.len);
        self.ord.swap(i, self.len);
        self.facts.swap(i, self.len);
        self.len -= 1;
    }

    fn drain_filter(&mut self, mut should_rm: impl FnMut(&I, &mut L, &mut F) -> bool) {
        let mut i = 0;
        while i < self.len {
            let (idx, l) = &mut self.ord[i];
            let f = &mut self.facts[i];
            if should_rm(idx, l, f) {
                self.remove_idx(i);
                continue;
            }
            i += 1;
        }
    }
}

impl<I: Idx, L: Ord + Eq + Copy, F: Eq, const N: usize> JoinSemiLattice for FactCache<I, L, F, N> {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        self.drain_filter(|i, l, f| {
            let Some((other_loc, other_fact)) = other.get(*i) else {
                changed = true;
                return true;
            };
            if other_fact == f {
                *l = (*l).max(*other_loc);
                return false;
            }
            changed = true;
            return true;
        });

        changed
    }
}
