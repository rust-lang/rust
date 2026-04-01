use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ops::Deref;

use rustc_index::bit_set::BitMatrix;

use crate::frozen::Frozen;
use crate::fx::{FxHashSet, FxIndexSet};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct TransitiveRelationBuilder<T> {
    // List of elements. This is used to map from a T to a usize.
    elements: FxIndexSet<T>,

    // List of base edges in the graph. Require to compute transitive
    // closure.
    edges: FxHashSet<Edge>,
}

#[derive(Debug)]
pub struct TransitiveRelation<T> {
    // Frozen transitive relation elements and edges.
    builder: Frozen<TransitiveRelationBuilder<T>>,

    // Cached transitive closure derived from the edges.
    closure: Frozen<BitMatrix<usize, usize>>,
}

impl<T> Deref for TransitiveRelation<T> {
    type Target = Frozen<TransitiveRelationBuilder<T>>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<T: Clone> Clone for TransitiveRelation<T> {
    fn clone(&self) -> Self {
        TransitiveRelation {
            builder: Frozen::freeze(self.builder.deref().clone()),
            closure: Frozen::freeze(self.closure.deref().clone()),
        }
    }
}

// HACK(eddyb) manual impl avoids `Default` bound on `T`.
impl<T: Eq + Hash> Default for TransitiveRelationBuilder<T> {
    fn default() -> Self {
        TransitiveRelationBuilder { elements: Default::default(), edges: Default::default() }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug, Hash)]
struct Index(usize);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
struct Edge {
    source: Index,
    target: Index,
}

impl<T: Eq + Hash + Copy> TransitiveRelationBuilder<T> {
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn elements(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    fn index(&self, a: T) -> Option<Index> {
        self.elements.get_index_of(&a).map(Index)
    }

    fn add_index(&mut self, a: T) -> Index {
        let (index, _added) = self.elements.insert_full(a);
        Index(index)
    }

    /// Applies the (partial) function to each edge and returns a new
    /// relation builder. If `f` returns `None` for any end-point,
    /// returns `None`.
    pub fn maybe_map<F, U>(&self, mut f: F) -> Option<TransitiveRelationBuilder<U>>
    where
        F: FnMut(T) -> Option<U>,
        U: Clone + Debug + Eq + Hash + Copy,
    {
        let mut result = TransitiveRelationBuilder::default();
        for edge in &self.edges {
            result.add(f(self.elements[edge.source.0])?, f(self.elements[edge.target.0])?);
        }
        Some(result)
    }

    /// Indicate that `a < b` (where `<` is this relation)
    pub fn add(&mut self, a: T, b: T) {
        let a = self.add_index(a);
        let b = self.add_index(b);
        let edge = Edge { source: a, target: b };
        self.edges.insert(edge);
    }

    /// Compute the transitive closure derived from the edges, and converted to
    /// the final result. After this, all elements will be immutable to maintain
    /// the correctness of the result.
    pub fn freeze(self) -> TransitiveRelation<T> {
        let mut matrix = BitMatrix::new(self.elements.len(), self.elements.len());
        let mut changed = true;
        while changed {
            changed = false;
            for edge in &self.edges {
                // add an edge from S -> T
                changed |= matrix.insert(edge.source.0, edge.target.0);

                // add all outgoing edges from T into S
                changed |= matrix.union_rows(edge.target.0, edge.source.0);
            }
        }
        TransitiveRelation { builder: Frozen::freeze(self), closure: Frozen::freeze(matrix) }
    }
}

impl<T: Eq + Hash + Copy> TransitiveRelation<T> {
    /// Applies the (partial) function to each edge and returns a new
    /// relation including transitive closures.
    pub fn maybe_map<F, U>(&self, f: F) -> Option<TransitiveRelation<U>>
    where
        F: FnMut(T) -> Option<U>,
        U: Clone + Debug + Eq + Hash + Copy,
    {
        Some(self.builder.maybe_map(f)?.freeze())
    }

    /// Checks whether `a < target` (transitively)
    pub fn contains(&self, a: T, b: T) -> bool {
        match (self.index(a), self.index(b)) {
            (Some(a), Some(b)) => self.with_closure(|closure| closure.contains(a.0, b.0)),
            (None, _) | (_, None) => false,
        }
    }

    /// Thinking of `x R y` as an edge `x -> y` in a graph, this
    /// returns all things reachable from `a`.
    ///
    /// Really this probably ought to be `impl Iterator<Item = &T>`, but
    /// I'm too lazy to make that work, and -- given the caching
    /// strategy -- it'd be a touch tricky anyhow.
    pub fn reachable_from(&self, a: T) -> Vec<T> {
        match self.index(a) {
            Some(a) => {
                self.with_closure(|closure| closure.iter(a.0).map(|i| self.elements[i]).collect())
            }
            None => vec![],
        }
    }

    /// Picks what I am referring to as the "postdominating"
    /// upper-bound for `a` and `b`. This is usually the least upper
    /// bound, but in cases where there is no single least upper
    /// bound, it is the "mutual immediate postdominator", if you
    /// imagine a graph where `a < b` means `a -> b`.
    ///
    /// This function is needed because region inference currently
    /// requires that we produce a single "UB", and there is no best
    /// choice for the LUB. Rather than pick arbitrarily, I pick a
    /// less good, but predictable choice. This should help ensure
    /// that region inference yields predictable results (though it
    /// itself is not fully sufficient).
    ///
    /// Examples are probably clearer than any prose I could write
    /// (there are corresponding tests below, btw). In each case,
    /// the query is `postdom_upper_bound(a, b)`:
    ///
    /// ```text
    /// // Returns Some(x), which is also LUB.
    /// a -> a1 -> x
    ///            ^
    ///            |
    /// b -> b1 ---+
    ///
    /// // Returns `Some(x)`, which is not LUB (there is none)
    /// // diagonal edges run left-to-right.
    /// a -> a1 -> x
    ///   \/       ^
    ///   /\       |
    /// b -> b1 ---+
    ///
    /// // Returns `None`.
    /// a -> a1
    /// b -> b1
    /// ```
    pub fn postdom_upper_bound(&self, a: T, b: T) -> Option<T> {
        let mubs = self.minimal_upper_bounds(a, b);
        self.mutual_immediate_postdominator(mubs)
    }

    /// Viewing the relation as a graph, computes the "mutual
    /// immediate postdominator" of a set of points (if one
    /// exists). See `postdom_upper_bound` for details.
    pub fn mutual_immediate_postdominator(&self, mut mubs: Vec<T>) -> Option<T> {
        loop {
            match mubs[..] {
                [] => return None,
                [mub] => return Some(mub),
                _ => {
                    let m = mubs.pop().unwrap();
                    let n = mubs.pop().unwrap();
                    mubs.extend(self.minimal_upper_bounds(n, m));
                }
            }
        }
    }

    /// Returns the set of bounds `X` such that:
    ///
    /// - `a < X` and `b < X`
    /// - there is no `Y != X` such that `a < Y` and `Y < X`
    ///   - except for the case where `X < a` (i.e., a strongly connected
    ///     component in the graph). In that case, the smallest
    ///     representative of the SCC is returned (as determined by the
    ///     internal indices).
    ///
    /// Note that this set can, in principle, have any size.
    pub fn minimal_upper_bounds(&self, a: T, b: T) -> Vec<T> {
        let (Some(mut a), Some(mut b)) = (self.index(a), self.index(b)) else {
            return vec![];
        };

        // in some cases, there are some arbitrary choices to be made;
        // it doesn't really matter what we pick, as long as we pick
        // the same thing consistently when queried, so ensure that
        // (a, b) are in a consistent relative order
        if a > b {
            mem::swap(&mut a, &mut b);
        }

        let lub_indices = self.with_closure(|closure| {
            // Easy case is when either a < b or b < a:
            if closure.contains(a.0, b.0) {
                return vec![b.0];
            }
            if closure.contains(b.0, a.0) {
                return vec![a.0];
            }

            // Otherwise, the tricky part is that there may be some c
            // where a < c and b < c. In fact, there may be many such
            // values. So here is what we do:
            //
            // 1. Find the vector `[X | a < X && b < X]` of all values
            //    `X` where `a < X` and `b < X`. In terms of the
            //    graph, this means all values reachable from both `a`
            //    and `b`. Note that this vector is also a set, but we
            //    use the term vector because the order matters
            //    to the steps below.
            //    - This vector contains upper bounds, but they are
            //      not minimal upper bounds. So you may have e.g.
            //      `[x, y, tcx, z]` where `x < tcx` and `y < tcx` and
            //      `z < x` and `z < y`:
            //
            //           z --+---> x ----+----> tcx
            //               |           |
            //               |           |
            //               +---> y ----+
            //
            //      In this case, we really want to return just `[z]`.
            //      The following steps below achieve this by gradually
            //      reducing the list.
            // 2. Pare down the vector using `pare_down`. This will
            //    remove elements from the vector that can be reached
            //    by an earlier element.
            //    - In the example above, this would convert `[x, y,
            //      tcx, z]` to `[x, y, z]`. Note that `x` and `y` are
            //      still in the vector; this is because while `z < x`
            //      (and `z < y`) holds, `z` comes after them in the
            //      vector.
            // 3. Reverse the vector and repeat the pare down process.
            //    - In the example above, we would reverse to
            //      `[z, y, x]` and then pare down to `[z]`.
            // 4. Reverse once more just so that we yield a vector in
            //    increasing order of index. Not necessary, but why not.
            //
            // I believe this algorithm yields a minimal set. The
            // argument is that, after step 2, we know that no element
            // can reach its successors (in the vector, not the graph).
            // After step 3, we know that no element can reach any of
            // its predecessors (because of step 2) nor successors
            // (because we just called `pare_down`)
            //
            // This same algorithm is used in `parents` below.

            let mut candidates = closure.intersect_rows(a.0, b.0); // (1)
            pare_down(&mut candidates, closure); // (2)
            candidates.reverse(); // (3a)
            pare_down(&mut candidates, closure); // (3b)
            candidates
        });

        lub_indices
            .into_iter()
            .rev() // (4)
            .map(|i| self.elements[i])
            .collect()
    }

    /// Given an element A, returns the maximal set {B} of elements B
    /// such that
    ///
    /// - A != B
    /// - A R B is true
    /// - for each i, j: `B[i]` R `B[j]` does not hold
    ///
    /// The intuition is that this moves "one step up" through a lattice
    /// (where the relation is encoding the `<=` relation for the lattice).
    /// So e.g., if the relation is `->` and we have
    ///
    /// ```text
    /// a -> b -> d -> f
    /// |              ^
    /// +--> c -> e ---+
    /// ```
    ///
    /// then `parents(a)` returns `[b, c]`. The `postdom_parent` function
    /// would further reduce this to just `f`.
    pub fn parents(&self, a: T) -> Vec<T> {
        let Some(a) = self.index(a) else {
            return vec![];
        };

        // Steal the algorithm for `minimal_upper_bounds` above, but
        // with a slight tweak. In the case where `a R a`, we remove
        // that from the set of candidates.
        let ancestors = self.with_closure(|closure| {
            let mut ancestors = closure.intersect_rows(a.0, a.0);

            // Remove anything that can reach `a`. If this is a
            // reflexive relation, this will include `a` itself.
            ancestors.retain(|&e| !closure.contains(e, a.0));

            pare_down(&mut ancestors, closure); // (2)
            ancestors.reverse(); // (3a)
            pare_down(&mut ancestors, closure); // (3b)
            ancestors
        });

        ancestors
            .into_iter()
            .rev() // (4)
            .map(|i| self.elements[i])
            .collect()
    }

    /// Given an element A, elements B with the lowest index such that `A R B`
    /// and `B R A`, or `A` if no such element exists.
    pub fn minimal_scc_representative(&self, a: T) -> T {
        match self.index(a) {
            Some(a_i) => self.with_closure(|closure| {
                closure
                    .iter(a_i.0)
                    .find(|i| closure.contains(*i, a_i.0))
                    .map_or(a, |i| self.elements[i])
            }),
            None => a,
        }
    }

    fn with_closure<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&BitMatrix<usize, usize>) -> R,
    {
        op(&self.closure)
    }

    /// Lists all the base edges in the graph: the initial _non-transitive_ set of element
    /// relations, which will be later used as the basis for the transitive closure computation.
    pub fn base_edges(&self) -> impl Iterator<Item = (T, T)> {
        self.edges
            .iter()
            .map(move |edge| (self.elements[edge.source.0], self.elements[edge.target.0]))
    }
}

/// Pare down is used as a step in the LUB computation. It edits the
/// candidates array in place by removing any element j for which
/// there exists an earlier element i<j such that i -> j. That is,
/// after you run `pare_down`, you know that for all elements that
/// remain in candidates, they cannot reach any of the elements that
/// come after them.
///
/// Examples follow. Assume that a -> b -> c and x -> y -> z.
///
/// - Input: `[a, b, x]`. Output: `[a, x]`.
/// - Input: `[b, a, x]`. Output: `[b, a, x]`.
/// - Input: `[a, x, b, y]`. Output: `[a, x]`.
fn pare_down(candidates: &mut Vec<usize>, closure: &BitMatrix<usize, usize>) {
    let mut i = 0;
    while let Some(&candidate_i) = candidates.get(i) {
        i += 1;

        let mut j = i;
        let mut dead = 0;
        while let Some(&candidate_j) = candidates.get(j) {
            if closure.contains(candidate_i, candidate_j) {
                // If `i` can reach `j`, then we can remove `j`. So just
                // mark it as dead and move on; subsequent indices will be
                // shifted into its place.
                dead += 1;
            } else {
                candidates[j - dead] = candidate_j;
            }
            j += 1;
        }
        candidates.truncate(j - dead);
    }
}
