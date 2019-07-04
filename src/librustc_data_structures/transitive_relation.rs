use crate::bit_set::BitMatrix;
use crate::fx::FxHashMap;
use crate::stable_hasher::{HashStable, StableHasher, StableHasherResult};
use crate::sync::Lock;
use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;


#[derive(Clone, Debug)]
pub struct TransitiveRelation<T: Clone + Debug + Eq + Hash> {
    // List of elements. This is used to map from a T to a usize.
    elements: Vec<T>,

    // Maps each element to an index.
    map: FxHashMap<T, Index>,

    // List of base edges in the graph. Require to compute transitive
    // closure.
    edges: Vec<Edge>,

    // This is a cached transitive closure derived from the edges.
    // Currently, we build it lazilly and just throw out any existing
    // copy whenever a new edge is added. (The Lock is to permit
    // the lazy computation.) This is kind of silly, except for the
    // fact its size is tied to `self.elements.len()`, so I wanted to
    // wait before building it up to avoid reallocating as new edges
    // are added with new elements. Perhaps better would be to ask the
    // user for a batch of edges to minimize this effect, but I
    // already wrote the code this way. :P -nmatsakis
    closure: Lock<Option<BitMatrix<usize, usize>>>,
}

// HACK(eddyb) manual impl avoids `Default` bound on `T`.
impl<T: Clone + Debug + Eq + Hash> Default for TransitiveRelation<T> {
    fn default() -> Self {
        TransitiveRelation {
            elements: Default::default(),
            map: Default::default(),
            edges: Default::default(),
            closure: Default::default(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable, Debug)]
struct Index(usize);

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
struct Edge {
    source: Index,
    target: Index,
}

impl<T: Clone + Debug + Eq + Hash> TransitiveRelation<T> {
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn elements(&self) -> impl Iterator<Item=&T> {
        self.elements.iter()
    }

    fn index(&self, a: &T) -> Option<Index> {
        self.map.get(a).cloned()
    }

    fn add_index(&mut self, a: T) -> Index {
        let &mut TransitiveRelation {
            ref mut elements,
            ref mut closure,
            ref mut map,
            ..
        } = self;

        *map.entry(a.clone())
           .or_insert_with(|| {
               elements.push(a);

               // if we changed the dimensions, clear the cache
               *closure.get_mut() = None;

               Index(elements.len() - 1)
           })
    }

    /// Applies the (partial) function to each edge and returns a new
    /// relation. If `f` returns `None` for any end-point, returns
    /// `None`.
    pub fn maybe_map<F, U>(&self, mut f: F) -> Option<TransitiveRelation<U>>
        where F: FnMut(&T) -> Option<U>,
              U: Clone + Debug + Eq + Hash + Clone,
    {
        let mut result = TransitiveRelation::default();
        for edge in &self.edges {
            result.add(f(&self.elements[edge.source.0])?, f(&self.elements[edge.target.0])?);
        }
        Some(result)
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
            *self.closure.get_mut() = None;
        }
    }

    /// Checks whether `a < target` (transitively)
    pub fn contains(&self, a: &T, b: &T) -> bool {
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
    pub fn reachable_from(&self, a: &T) -> Vec<&T> {
        match self.index(a) {
            Some(a) => self.with_closure(|closure| {
                closure.iter(a.0).map(|i| &self.elements[i]).collect()
            }),
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
    pub fn postdom_upper_bound(&self, a: &T, b: &T) -> Option<&T> {
        let mubs = self.minimal_upper_bounds(a, b);
        self.mutual_immediate_postdominator(mubs)
    }

    /// Viewing the relation as a graph, computes the "mutual
    /// immediate postdominator" of a set of points (if one
    /// exists). See `postdom_upper_bound` for details.
    pub fn mutual_immediate_postdominator<'a>(&'a self, mut mubs: Vec<&'a T>) -> Option<&'a T> {
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
            //
            // This same algorithm is used in `parents` below.

            let mut candidates = closure.intersect_rows(a.0, b.0); // (1)
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

    /// Given an element A, returns the maximal set {B} of elements B
    /// such that
    ///
    /// - A != B
    /// - A R B is true
    /// - for each i, j: B[i] R B[j] does not hold
    ///
    /// The intuition is that this moves "one step up" through a lattice
    /// (where the relation is encoding the `<=` relation for the lattice).
    /// So e.g., if the relation is `->` and we have
    ///
    /// ```
    /// a -> b -> d -> f
    /// |              ^
    /// +--> c -> e ---+
    /// ```
    ///
    /// then `parents(a)` returns `[b, c]`. The `postdom_parent` function
    /// would further reduce this to just `f`.
    pub fn parents(&self, a: &T) -> Vec<&T> {
        let a = match self.index(a) {
            Some(a) => a,
            None => return vec![]
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

        ancestors.into_iter()
                 .rev() // (4)
                 .map(|i| &self.elements[i])
                 .collect()
    }

    /// A "best" parent in some sense. See `parents` and
    /// `postdom_upper_bound` for more details.
    pub fn postdom_parent(&self, a: &T) -> Option<&T> {
        self.mutual_immediate_postdominator(self.parents(a))
    }

    fn with_closure<OP, R>(&self, op: OP) -> R
        where OP: FnOnce(&BitMatrix<usize, usize>) -> R
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

    fn compute_closure(&self) -> BitMatrix<usize, usize> {
        let mut matrix = BitMatrix::new(self.elements.len(),
                                        self.elements.len());
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
fn pare_down(candidates: &mut Vec<usize>, closure: &BitMatrix<usize, usize>) {
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

impl<T> Encodable for TransitiveRelation<T>
    where T: Clone + Encodable + Debug + Eq + Hash + Clone
{
    fn encode<E: Encoder>(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct("TransitiveRelation", 2, |s| {
            s.emit_struct_field("elements", 0, |s| self.elements.encode(s))?;
            s.emit_struct_field("edges", 1, |s| self.edges.encode(s))?;
            Ok(())
        })
    }
}

impl<T> Decodable for TransitiveRelation<T>
    where T: Clone + Decodable + Debug + Eq + Hash + Clone
{
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        d.read_struct("TransitiveRelation", 2, |d| {
            let elements: Vec<T> = d.read_struct_field("elements", 0, |d| Decodable::decode(d))?;
            let edges = d.read_struct_field("edges", 1, |d| Decodable::decode(d))?;
            let map = elements.iter()
                              .enumerate()
                              .map(|(index, elem)| (elem.clone(), Index(index)))
                              .collect();
            Ok(TransitiveRelation { elements, edges, map, closure: Lock::new(None) })
        })
    }
}

impl<CTX, T> HashStable<CTX> for TransitiveRelation<T>
    where T: HashStable<CTX> + Eq + Debug + Clone + Hash
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        // We are assuming here that the relation graph has been built in a
        // deterministic way and we can just hash it the way it is.
        let TransitiveRelation {
            ref elements,
            ref edges,
            // "map" is just a copy of elements vec
            map: _,
            // "closure" is just a copy of the data above
            closure: _
        } = *self;

        elements.hash_stable(hcx, hasher);
        edges.hash_stable(hcx, hasher);
    }
}

impl<CTX> HashStable<CTX> for Edge {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let Edge {
            ref source,
            ref target,
        } = *self;

        source.hash_stable(hcx, hasher);
        target.hash_stable(hcx, hasher);
    }
}

impl<CTX> HashStable<CTX> for Index {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let Index(idx) = *self;
        idx.hash_stable(hcx, hasher);
    }
}

#[test]
fn test_one_step() {
    let mut relation = TransitiveRelation::default();
    relation.add("a", "b");
    relation.add("a", "c");
    assert!(relation.contains(&"a", &"c"));
    assert!(relation.contains(&"a", &"b"));
    assert!(!relation.contains(&"b", &"a"));
    assert!(!relation.contains(&"a", &"d"));
}

#[test]
fn test_many_steps() {
    let mut relation = TransitiveRelation::default();
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
fn mubs_triangle() {
    // a -> tcx
    //      ^
    //      |
    //      b
    let mut relation = TransitiveRelation::default();
    relation.add("a", "tcx");
    relation.add("b", "tcx");
    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"tcx"]);
    assert_eq!(relation.parents(&"a"), vec![&"tcx"]);
    assert_eq!(relation.parents(&"b"), vec![&"tcx"]);
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

    let mut relation = TransitiveRelation::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"2"]);
    assert_eq!(relation.parents(&"0"), vec![&"2"]);
    assert_eq!(relation.parents(&"2"), vec![&"1"]);
    assert!(relation.parents(&"1").is_empty());
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

    let mut relation = TransitiveRelation::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1"]);
    assert_eq!(relation.parents(&"0"), vec![&"1"]);
    assert_eq!(relation.parents(&"1"), vec![&"2"]);
    assert!(relation.parents(&"2").is_empty());
}

#[test]
fn mubs_no_best_choice() {
    // in this case, the intersection yields [1, 2], and the "pare
    // down" calls find nothing to remove.
    let mut relation = TransitiveRelation::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1", &"2"]);
    assert_eq!(relation.parents(&"0"), vec![&"1", &"2"]);
    assert_eq!(relation.parents(&"3"), vec![&"1", &"2"]);
}

#[test]
fn mubs_best_choice_scc() {
    // in this case, 1 and 2 form a cycle; we pick arbitrarily (but
    // consistently).

    let mut relation = TransitiveRelation::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");
    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");

    assert_eq!(relation.minimal_upper_bounds(&"0", &"3"), vec![&"1"]);
    assert_eq!(relation.parents(&"0"), vec![&"1"]);
}

#[test]
fn pdub_crisscross() {
    // diagonal edges run left-to-right
    // a -> a1 -> x
    //   \/       ^
    //   /\       |
    // b -> b1 ---+

    let mut relation = TransitiveRelation::default();
    relation.add("a", "a1");
    relation.add("a", "b1");
    relation.add("b", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"),
               vec![&"a1", &"b1"]);
    assert_eq!(relation.postdom_upper_bound(&"a", &"b"), Some(&"x"));
    assert_eq!(relation.postdom_parent(&"a"), Some(&"x"));
    assert_eq!(relation.postdom_parent(&"b"), Some(&"x"));
}

#[test]
fn pdub_crisscross_more() {
    // diagonal edges run left-to-right
    // a -> a1 -> a2 -> a3 -> x
    //   \/    \/             ^
    //   /\    /\             |
    // b -> b1 -> b2 ---------+

    let mut relation = TransitiveRelation::default();
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

    assert_eq!(relation.postdom_parent(&"a"), Some(&"x"));
    assert_eq!(relation.postdom_parent(&"b"), Some(&"x"));
}

#[test]
fn pdub_lub() {
    // a -> a1 -> x
    //            ^
    //            |
    // b -> b1 ---+

    let mut relation = TransitiveRelation::default();
    relation.add("a", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"x"]);
    assert_eq!(relation.postdom_upper_bound(&"a", &"b"), Some(&"x"));

    assert_eq!(relation.postdom_parent(&"a"), Some(&"a1"));
    assert_eq!(relation.postdom_parent(&"b"), Some(&"b1"));
    assert_eq!(relation.postdom_parent(&"a1"), Some(&"x"));
    assert_eq!(relation.postdom_parent(&"b1"), Some(&"x"));
}

#[test]
fn mubs_intermediate_node_on_one_side_only() {
    // a -> c -> d
    //           ^
    //           |
    //           b

    // "digraph { a -> c -> d; b -> d; }",
    let mut relation = TransitiveRelation::default();
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
    let mut relation = TransitiveRelation::default();
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
    let mut relation = TransitiveRelation::default();
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
    let mut relation = TransitiveRelation::default();
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
    let mut relation = TransitiveRelation::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "e");
    relation.add("e", "c");
    relation.add("a", "d");
    relation.add("b", "e");

    assert_eq!(relation.minimal_upper_bounds(&"a", &"b"), vec![&"c"]);
}

#[test]
fn parent() {
    // An example that was misbehaving in the compiler.
    //
    // 4 -> 1 -> 3
    //   \  |   /
    //    \ v  /
    // 2 -> 0
    //
    // plus a bunch of self-loops
    //
    // Here `->` represents `<=` and `0` is `'static`.

    let pairs = vec![
        (2, /*->*/ 0),
        (2, /*->*/ 2),
        (0, /*->*/ 0),
        (0, /*->*/ 0),
        (1, /*->*/ 0),
        (1, /*->*/ 1),
        (3, /*->*/ 0),
        (3, /*->*/ 3),
        (4, /*->*/ 0),
        (4, /*->*/ 1),
        (1, /*->*/ 3),
    ];

    let mut relation = TransitiveRelation::default();
    for (a, b) in pairs {
        relation.add(a, b);
    }

    let p = relation.postdom_parent(&3);
    assert_eq!(p, Some(&0));
}
