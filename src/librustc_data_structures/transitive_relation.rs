// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use bitvec::BitMatrix;
use std::cell::RefCell;
use std::fmt::Debug;
use std::mem;

#[derive(Clone)]
pub struct TransitiveRelation<T: Debug + PartialEq> {
    // List of elements. This is used to map from a T to a usize.  We
    // expect domain to be small so just use a linear list versus a
    // hashmap or something.
    elements: Vec<T>,

    // List of base edges in the graph. Require to compute transitive
    // closure.
    edges: Vec<Edge>,

    // This is a cached transitive closure derived from the edges.
    // Currently, we build it lazilly and just throw out any existing
    // copy whenever a new edge is added. (The RefCell is to permit
    // the lazy computation.) This is kind of silly, except for the
    // fact its size is tied to `self.elements.len()`, so I wanted to
    // wait before building it up to avoid reallocating as new edges
    // are added with new elements. Perhaps better would be to ask the
    // user for a batch of edges to minimize this effect, but I
    // already wrote the code this way. :P -nmatsakis
    closure: RefCell<Option<BitMatrix>>,
}

#[derive(Clone, PartialEq, PartialOrd)]
struct Index(usize);

#[derive(Clone, PartialEq)]
struct Edge {
    source: Index,
    target: Index,
}

impl<T: Debug + PartialEq> TransitiveRelation<T> {
    pub fn new() -> TransitiveRelation<T> {
        TransitiveRelation {
            elements: vec![],
            edges: vec![],
            closure: RefCell::new(None),
        }
    }

    fn index(&self, a: &T) -> Option<Index> {
        self.elements.iter().position(|e| *e == *a).map(Index)
    }

    fn add_index(&mut self, a: T) -> Index {
        match self.index(&a) {
            Some(i) => i,
            None => {
                self.elements.push(a);

                // if we changed the dimensions, clear the cache
                *self.closure.borrow_mut() = None;

                Index(self.elements.len() - 1)
            }
        }
    }

    /// Indicate that `a < b` (where `<` is this relation)
    pub fn add(&mut self, a: T, b: T) {
        let a = self.add_index(a);
        let b = self.add_index(b);
        let edge = Edge {
            source: a,
            target: b,
        };
        if !self.edges.contains(&edge) {
            self.edges.push(edge);

            // added an edge, clear the cache
            *self.closure.borrow_mut() = None;
        }
    }

    /// Check whether `a < target` (transitively)
    pub fn contains(&self, a: &T, b: &T) -> bool {
        match (self.index(a), self.index(b)) {
            (Some(a), Some(b)) => self.with_closure(|closure| closure.contains(a.0, b.0)),
            (None, _) | (_, None) => false,
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
    /// // returns Some(x), which is also LUB
    /// a -> a1 -> x
    ///            ^
    ///            |
    /// b -> b1 ---+
    ///
    /// // returns Some(x), which is not LUB (there is none)
    /// // diagonal edges run left-to-right
    /// a -> a1 -> x
    ///   \/       ^
    ///   /\       |
    /// b -> b1 ---+
    ///
    /// // returns None
    /// a -> a1
    /// b -> b1
    /// ```
    pub fn postdom_upper_bound(&self, a: &T, b: &T) -> Option<&T> {
        let mut mubs = self.minimal_upper_bounds(a, b);
        loop {
            match mubs.len() {
                0 => return None,
                1 => return Some(mubs[0]),
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
    pub fn minimal_upper_bounds(&self, a: &T, b: &T) -> Vec<&T> {
        let (mut a, mut b) = match (self.index(a), self.index(b)) {
            (Some(a), Some(b)) => (a, b),
            (None, _) | (_, None) => {
                return vec![];
            }
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
            //    `X` where `a < X` and `b < X`.  In terms of the
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
            // its predecesssors (because of step 2) nor successors
            // (because we just called `pare_down`)

            let mut candidates = closure.intersection(a.0, b.0); // (1)
            pare_down(&mut candidates, closure); // (2)
            candidates.reverse(); // (3a)
            pare_down(&mut candidates, closure); // (3b)
            candidates
        });

        lub_indices.into_iter()
                   .rev() // (4)
                   .map(|i| &self.elements[i])
                   .collect()
    }

    fn with_closure<OP, R>(&self, op: OP) -> R
        where OP: FnOnce(&BitMatrix) -> R
    {
        let mut closure_cell = self.closure.borrow_mut();
        let mut closure = closure_cell.take();
        if closure.is_none() {
            closure = Some(self.compute_closure());
        }
        let result = op(closure.as_ref().unwrap());
        *closure_cell = closure;
        result
    }

    fn compute_closure(&self) -> BitMatrix {
        let mut matrix = BitMatrix::new(self.elements.len(),
                                        self.elements.len());
        let mut changed = true;
        while changed {
            changed = false;
            for edge in self.edges.iter() {
                // add an edge from S -> T
                changed |= matrix.add(edge.source.0, edge.target.0);

                // add all outgoing edges from T into S
                changed |= matrix.merge(edge.target.0, edge.source.0);
            }
        }
        matrix
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
fn pare_down(candidates: &mut Vec<usize>, closure: &BitMatrix) {
    let mut i = 0;
    while i < candidates.len() {
        let candidate_i = candidates[i];
        i += 1;

        let mut j = i;
        let mut dead = 0;
        while j < candidates.len() {
            let candidate_j = candidates[j];
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

#[test]
fn test_one_step() {
    let mut relation = TransitiveRelation::new();
    relation.add("a", "b");
    relation.add("a", "c");
    assert!(relation.contains(&"a", &"c"));
    assert!(relation.contains(&"a", &"b"));
    assert!(!relation.contains(&"b", &"a"));
    assert!(!relation.contains(&"a", &"d"));
}

#[test]
fn test_many_steps() {
    let mut relation = TransitiveRelation::new();
    relation.add("a", "b");
    relation.add("a", "c");
    relation.add("a", "f");

    relation.add("b", "c");
    relation.add("b", "d");
    relation.add("b", "e");

    relation.add("e", "g");

    assert!(relation.contains(&"a", &"b"));
    assert!(relation.contains(&"a", &"c"));
    assert!(relation.contains(&"a", &"d"));
    assert!(relation.contains(&"a", &"e"));
    assert!(relation.contains(&"a", &"f"));
    assert!(relation.contains(&"a", &"g"));

    assert!(relation.contains(&"b", &"g"));

    assert!(!relation.contains(&"a", &"x"));
    assert!(!relation.contains(&"b", &"f"));
}

#[test]
fn mubs_triange() {
    let mut relation = TransitiveRelation::new();
    relation.add("a", "tcx");
    relation.add("b", "tcx");
    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"tcx"]);
}

#[test]
fn mubs_best_choice1() {
    // 0 -> 1 <- 3
    // |    ^    |
    // |    |    |
    // +--> 2 <--+
    //
    // mubs(0,3) = [1]

    // This tests a particular state in the algorithm, in which we
    // need the second pare down call to get the right result (after
    // intersection, we have [1, 2], but 2 -> 1).

    let mut relation = TransitiveRelation::new();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"2"]);
}

#[test]
fn mubs_best_choice2() {
    // 0 -> 1 <- 3
    // |    |    |
    // |    v    |
    // +--> 2 <--+
    //
    // mubs(0,3) = [2]

    // Like the precedecing test, but in this case intersection is [2,
    // 1], and hence we rely on the first pare down call.

    let mut relation = TransitiveRelation::new();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1"]);
}

#[test]
fn mubs_no_best_choice() {
    // in this case, the intersection yields [1, 2], and the "pare
    // down" calls find nothing to remove.
    let mut relation = TransitiveRelation::new();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1", &"2"]);
}

#[test]
fn mubs_best_choice_scc() {
    let mut relation = TransitiveRelation::new();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");
    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1"]);
}

#[test]
fn pdub_crisscross() {
    // diagonal edges run left-to-right
    // a -> a1 -> x
    //   \/       ^
    //   /\       |
    // b -> b1 ---+

    let mut relation = TransitiveRelation::new();
    relation.add("a", "a1");
    relation.add("a", "b1");
    relation.add("b", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"),
               vec![&"a1", &"b1"]);
    assert_eq!(relation.postdom_upper_bound(&"a", &"b"), Some(&"x"));
}

#[test]
fn pdub_crisscross_more() {
    // diagonal edges run left-to-right
    // a -> a1 -> a2 -> a3 -> x
    //   \/    \/             ^
    //   /\    /\             |
    // b -> b1 -> b2 ---------+

    let mut relation = TransitiveRelation::new();
    relation.add("a", "a1");
    relation.add("a", "b1");
    relation.add("b", "a1");
    relation.add("b", "b1");

    relation.add("a1", "a2");
    relation.add("a1", "b2");
    relation.add("b1", "a2");
    relation.add("b1", "b2");

    relation.add("a2", "a3");

    relation.add("a3", "x");
    relation.add("b2", "x");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"),
               vec![&"a1", &"b1"]);
    assert_eq!(relation.minimal_upper_bounds(&"a1", &"b1"),
               vec![&"a2", &"b2"]);
    assert_eq!(relation.postdom_upper_bound(&"a", &"b"), Some(&"x"));
}

#[test]
fn pdub_lub() {
    // a -> a1 -> x
    //            ^
    //            |
    // b -> b1 ---+

    let mut relation = TransitiveRelation::new();
    relation.add("a", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"x"]);
    assert_eq!(relation.postdom_upper_bound(&"a", &"b"), Some(&"x"));
}

#[test]
fn mubs_intermediate_node_on_one_side_only() {
    // a -> c -> d
    //           ^
    //           |
    //           b

    // "digraph { a -> c -> d; b -> d; }",
    let mut relation = TransitiveRelation::new();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("b", "d");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"d"]);
}

#[test]
fn mubs_scc_1() {
    // +-------------+
    // |    +----+   |
    // |    v    |   |
    // a -> c -> d <-+
    //           ^
    //           |
    //           b

    // "digraph { a -> c -> d; d -> c; a -> d; b -> d; }",
    let mut relation = TransitiveRelation::new();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "c");
    relation.add("a", "d");
    relation.add("b", "d");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"c"]);
}

#[test]
fn mubs_scc_2() {
    //      +----+
    //      v    |
    // a -> c -> d
    //      ^    ^
    //      |    |
    //      +--- b

    // "digraph { a -> c -> d; d -> c; b -> d; b -> c; }",
    let mut relation = TransitiveRelation::new();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "c");
    relation.add("b", "d");
    relation.add("b", "c");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"c"]);
}

#[test]
fn mubs_scc_3() {
    //      +---------+
    //      v         |
    // a -> c -> d -> e
    //           ^    ^
    //           |    |
    //           b ---+

    // "digraph { a -> c -> d -> e -> c; b -> d; b -> e; }",
    let mut relation = TransitiveRelation::new();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "e");
    relation.add("e", "c");
    relation.add("b", "d");
    relation.add("b", "e");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"c"]);
}

#[test]
fn mubs_scc_4() {
    //      +---------+
    //      v         |
    // a -> c -> d -> e
    // |         ^    ^
    // +---------+    |
    //                |
    //           b ---+

    // "digraph { a -> c -> d -> e -> c; a -> d; b -> e; }"
    let mut relation = TransitiveRelation::new();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "e");
    relation.add("e", "c");
    relation.add("a", "d");
    relation.add("b", "e");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"c"]);
}
