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
//! and [`MeetSemiLattice`] do not have [`PartialOrd`] as a supertrait. This
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
use rustc_index::{Idx, IndexVec};
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

/// A set that has a "bottom" element, which is less than or equal to any other element.
pub trait HasBottom {
    const BOTTOM: Self;
}

/// A set that has a "top" element, which is greater than or equal to any other element.
pub trait HasTop {
    const TOP: Self;
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

impl HasBottom for bool {
    const BOTTOM: Self = false;
}

impl HasTop for bool {
    const TOP: Self = true;
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

impl<T> HasBottom for FlatSet<T> {
    const BOTTOM: Self = Self::Bottom;
}

impl<T> HasTop for FlatSet<T> {
    const TOP: Self = Self::Top;
}

/// Extend a lattice with a bottom value to represent an unreachable execution.
///
/// The only useful action on an unreachable state is joining it with a reachable one to make it
/// reachable. All other actions, gen/kill for instance, are no-ops.
#[derive(PartialEq, Eq, Debug)]
pub enum MaybeReachable<T> {
    Unreachable,
    Reachable(T),
}

impl<T> MaybeReachable<T> {
    pub fn is_reachable(&self) -> bool {
        matches!(self, MaybeReachable::Reachable(_))
    }
}

impl<T> HasBottom for MaybeReachable<T> {
    const BOTTOM: Self = MaybeReachable::Unreachable;
}

impl<T: HasTop> HasTop for MaybeReachable<T> {
    const TOP: Self = MaybeReachable::Reachable(T::TOP);
}

impl<S> MaybeReachable<S> {
    /// Return whether the current state contains the given element. If the state is unreachable,
    /// it does no contain anything.
    pub fn contains<T>(&self, elem: T) -> bool
    where
        S: BitSetExt<T>,
    {
        match self {
            MaybeReachable::Unreachable => false,
            MaybeReachable::Reachable(set) => set.contains(elem),
        }
    }
}

impl<T, S: BitSetExt<T>> BitSetExt<T> for MaybeReachable<S> {
    fn contains(&self, elem: T) -> bool {
        self.contains(elem)
    }

    fn union(&mut self, other: &HybridBitSet<T>) {
        match self {
            MaybeReachable::Unreachable => {}
            MaybeReachable::Reachable(set) => set.union(other),
        }
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) {
        match self {
            MaybeReachable::Unreachable => {}
            MaybeReachable::Reachable(set) => set.subtract(other),
        }
    }
}

impl<V: Clone> Clone for MaybeReachable<V> {
    fn clone(&self) -> Self {
        match self {
            MaybeReachable::Reachable(x) => MaybeReachable::Reachable(x.clone()),
            MaybeReachable::Unreachable => MaybeReachable::Unreachable,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (&mut *self, source) {
            (MaybeReachable::Reachable(x), MaybeReachable::Reachable(y)) => {
                x.clone_from(&y);
            }
            _ => *self = source.clone(),
        }
    }
}

impl<T: JoinSemiLattice + Clone> JoinSemiLattice for MaybeReachable<T> {
    fn join(&mut self, other: &Self) -> bool {
        // Unreachable acts as a bottom.
        match (&mut *self, &other) {
            (_, MaybeReachable::Unreachable) => false,
            (MaybeReachable::Unreachable, _) => {
                *self = other.clone();
                true
            }
            (MaybeReachable::Reachable(this), MaybeReachable::Reachable(other)) => this.join(other),
        }
    }
}
