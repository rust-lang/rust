use rustc_index::vec::{Idx, IndexVec};

use std::cell::Cell;

/// The maximum rank in a union-by-rank disjoint-set is `log N`, where `N` is the cardinality of
/// the set. The maximum value of an `Idx` is slightly less than 2^32, so we only need 5 bits to
/// store the rank.
type Rank = u8;

/// A [disjoint-set] (AKA union-find) data structure.
///
/// This stores a partition of a set of some index type `T` into disjoint subsets.
///
/// [disjoint-set]: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
//
// We use a structure of arrays here because `ranks` is rarely accessed (never in *Find*, exactly
// twice in *Union*) compared to `parents`.
#[derive(Clone)]
pub struct DisjointSet<T: Idx> {
    parents: IndexVec<T, Cell<T>>,
    ranks: IndexVec<T, Rank>,
}

impl<T: Idx> DisjointSet<T> {
    /// Returns a new `DisjointSet` where each element of `set` is in its own subset.
    pub fn new<E>(set: &IndexVec<T, E>) -> Self {
        DisjointSet::with_cardinality(set.len())
    }

    /// Returns a new `DisjointSet` for a set with cardinality `n` where each element is in its own
    /// subset.
    pub fn with_cardinality(n: usize) -> Self {
        DisjointSet {
            parents: IndexVec::from_fn_n(Cell::new, n),
            ranks: IndexVec::from_elem_n(0, n),
        }
    }

    /// Returns `true` if `x` and `y` are in the same subset.
    pub fn is_joint(&self, x: T, y: T) -> bool {
        self.find(x) == self.find(y)
    }

    /// Returns `true` if `x` and `y` are in different subsets.
    pub fn is_disjoint(&self, x: T, y: T) -> bool {
        !self.is_joint(x, y)
    }

    fn find(&self, x: T) -> T {
        find_elem_and_parent(&self.parents, x).elem
    }

    /// Merges the subset containing `x` with the subset containing `y`.
    pub fn union(&mut self, x: T, y: T) {
        let x = find_elem_and_parent(&self.parents, x);
        let y = find_elem_and_parent(&self.parents, y);

        if x.elem == y.elem {
            return;
        }

        let y_rank = self.ranks[y.elem];
        let x_rank = &mut self.ranks[x.elem];

        // The subset with the larger rank becomes the root node for the newly merged set.
        if *x_rank < y_rank {
            x.parent.set(y.elem);
        } else {
            y.parent.set(x.elem);
        }

        // If both subsets have equal rank, `x` is picked arbitrarily as the new root (see above)
        // and its rank is increased.
        if *x_rank == y_rank {
            *x_rank += 1;
        }
    }
}

/// An element of the disjoint-set forest and a reference to its entry in the `parents` array.
#[derive(Clone, Copy)]
struct ElemAndParent<'a, T> {
    elem: T,
    parent: &'a Cell<T>,
}

/// Run the *Find* operation for `elem`, performing path compression along the way.
fn find_elem_and_parent<T: Idx>(parents: &IndexVec<T, Cell<T>>, elem: T) -> ElemAndParent<'_, T> {
    let mut curr = ElemAndParent { elem, parent: &parents[elem] };

    loop {
        let parent = curr.parent.get();
        if parent == curr.elem {
            break;
        }

        let parent_of_parent = &parents[parent];

        // Do "path compression" by making our grandparent in the disjoint-set forest our new
        // parent. This reduces the length of the path from each node to the root of its
        // subtree.
        let grandparent = parent_of_parent.get();
        curr.parent.set(grandparent);

        // Recurse into our old parent node. This variant of path compression is called "path
        // splitting".
        curr.elem = parent;
        curr.parent = parent_of_parent;

        // Alternatively, we could recurse into our grandparent node. This variant of path
        // compression is called "path halving", and is slightly less eager.
        //
        //curr.elem = grandparent;
        //curr.parent = &self.parents[grandparent];
    }

    return curr;
}

#[cfg(test)]
mod tests;
