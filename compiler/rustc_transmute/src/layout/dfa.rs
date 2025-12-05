use std::fmt;
use std::iter::Peekable;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Byte, Reference, Region, Tree, Type, Uninhabited};
use crate::{Map, Set};

#[derive(PartialEq)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct Dfa<R, T>
where
    R: Region,
    T: Type,
{
    pub(crate) transitions: Map<State, Transitions<R, T>>,
    pub(crate) start: State,
    pub(crate) accept: State,
}

#[derive(PartialEq, Clone, Debug)]
pub(crate) struct Transitions<R, T>
where
    R: Region,
    T: Type,
{
    byte_transitions: EdgeSet<State>,
    ref_transitions: Map<Reference<R, T>, State>,
}

impl<R, T> Default for Transitions<R, T>
where
    R: Region,
    T: Type,
{
    fn default() -> Self {
        Self { byte_transitions: EdgeSet::empty(), ref_transitions: Map::default() }
    }
}

/// The states in a [`Dfa`] represent byte offsets.
#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Copy, Clone)]
pub(crate) struct State(pub(crate) u32);

impl State {
    pub(crate) fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S_{}", self.0)
    }
}

impl<R, T> Dfa<R, T>
where
    R: Region,
    T: Type,
{
    #[cfg(test)]
    pub(crate) fn bool() -> Self {
        Self::from_transitions(|accept| Transitions {
            byte_transitions: EdgeSet::new(Byte::new(0x00..=0x01), accept),
            ref_transitions: Map::default(),
        })
    }

    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Transitions<R, T>> = Map::default();
        let start = State::new();
        let accept = start;

        Self { transitions, start, accept }
    }

    pub(crate) fn from_byte(byte: Byte) -> Self {
        Self::from_transitions(|accept| Transitions {
            byte_transitions: EdgeSet::new(byte, accept),
            ref_transitions: Map::default(),
        })
    }

    pub(crate) fn from_ref(r: Reference<R, T>) -> Self {
        Self::from_transitions(|accept| Transitions {
            byte_transitions: EdgeSet::empty(),
            ref_transitions: [(r, accept)].into_iter().collect(),
        })
    }

    fn from_transitions(f: impl FnOnce(State) -> Transitions<R, T>) -> Self {
        let start = State::new();
        let accept = State::new();

        Self { transitions: [(start, f(accept))].into_iter().collect(), start, accept }
    }

    pub(crate) fn from_tree(tree: Tree<!, R, T>) -> Result<Self, Uninhabited> {
        Ok(match tree {
            Tree::Byte(b) => Self::from_byte(b),
            Tree::Ref(r) => Self::from_ref(r),
            Tree::Alt(alts) => {
                // Convert and filter the inhabited alternatives.
                let mut alts = alts.into_iter().map(Self::from_tree).filter_map(Result::ok);
                // If there are no alternatives, return `Uninhabited`.
                let dfa = alts.next().ok_or(Uninhabited)?;
                // Combine the remaining alternatives with `dfa`.
                alts.fold(dfa, |dfa, alt| dfa.union(alt, State::new))
            }
            Tree::Seq(elts) => {
                let mut dfa = Self::unit();
                for elt in elts.into_iter().map(Self::from_tree) {
                    dfa = dfa.concat(elt?);
                }
                dfa
            }
        })
    }

    /// Concatenate two `Dfa`s.
    pub(crate) fn concat(self, other: Self) -> Self {
        if self.start == self.accept {
            return other;
        } else if other.start == other.accept {
            return self;
        }

        let start = self.start;
        let accept = other.accept;

        let mut transitions: Map<State, Transitions<R, T>> = self.transitions;

        for (source, transition) in other.transitions {
            let fix_state = |state| if state == other.start { self.accept } else { state };
            let byte_transitions = transition.byte_transitions.map_states(&fix_state);
            let ref_transitions = transition
                .ref_transitions
                .into_iter()
                .map(|(r, state)| (r, fix_state(state)))
                .collect();

            let old = transitions
                .insert(fix_state(source), Transitions { byte_transitions, ref_transitions });
            assert!(old.is_none());
        }

        Self { transitions, start, accept }
    }

    /// Compute the union of two `Dfa`s.
    pub(crate) fn union(self, other: Self, mut new_state: impl FnMut() -> State) -> Self {
        // We implement `union` by lazily initializing a set of states
        // corresponding to the product of states in `self` and `other`, and
        // then add transitions between these states that correspond to where
        // they exist between `self` and `other`.

        let a = self;
        let b = other;

        let accept = new_state();

        let mut mapping: Map<(Option<State>, Option<State>), State> = Map::default();

        let mut mapped = |(a_state, b_state)| {
            if Some(a.accept) == a_state || Some(b.accept) == b_state {
                // If either `a_state` or `b_state` are accepting, map to a
                // common `accept` state.
                accept
            } else {
                *mapping.entry((a_state, b_state)).or_insert_with(&mut new_state)
            }
        };

        let start = mapped((Some(a.start), Some(b.start)));
        let mut transitions: Map<State, Transitions<R, T>> = Map::default();
        let empty_transitions = Transitions::default();

        struct WorkQueue {
            queue: Vec<(Option<State>, Option<State>)>,
            // Track all entries ever enqueued to avoid duplicating work. This
            // gives us a guarantee that a given (a_state, b_state) pair will
            // only ever be visited once.
            enqueued: Set<(Option<State>, Option<State>)>,
        }
        impl WorkQueue {
            fn enqueue(&mut self, a_state: Option<State>, b_state: Option<State>) {
                if self.enqueued.insert((a_state, b_state)) {
                    self.queue.push((a_state, b_state));
                }
            }
        }
        let mut queue = WorkQueue { queue: Vec::new(), enqueued: Set::default() };
        queue.enqueue(Some(a.start), Some(b.start));

        while let Some((a_src, b_src)) = queue.queue.pop() {
            let src = mapped((a_src, b_src));
            if src == accept {
                // While it's possible to have a DFA whose accept state has
                // out-edges, these do not affect the semantics of the DFA, and
                // so there's no point in processing them. Continuing here also
                // has the advantage of guaranteeing that we only ever process a
                // given node in the output DFA once. In particular, with the
                // exception of the accept state, we ensure that we only push a
                // given node to the `queue` once. This allows the following
                // code to assume that we're processing a node we've never
                // processed before, which means we never need to merge two edge
                // sets - we only ever need to construct a new edge set from
                // whole cloth.
                continue;
            }

            let a_transitions =
                a_src.and_then(|a_src| a.transitions.get(&a_src)).unwrap_or(&empty_transitions);
            let b_transitions =
                b_src.and_then(|b_src| b.transitions.get(&b_src)).unwrap_or(&empty_transitions);

            let byte_transitions = a_transitions.byte_transitions.union(
                &b_transitions.byte_transitions,
                |a_dst, b_dst| {
                    assert!(a_dst.is_some() || b_dst.is_some());

                    queue.enqueue(a_dst, b_dst);
                    mapped((a_dst, b_dst))
                },
            );

            let ref_transitions =
                a_transitions.ref_transitions.keys().chain(b_transitions.ref_transitions.keys());

            let ref_transitions = ref_transitions
                .map(|ref_transition| {
                    let a_dst = a_transitions.ref_transitions.get(ref_transition).copied();
                    let b_dst = b_transitions.ref_transitions.get(ref_transition).copied();

                    assert!(a_dst.is_some() || b_dst.is_some());

                    queue.enqueue(a_dst, b_dst);
                    (*ref_transition, mapped((a_dst, b_dst)))
                })
                .collect();

            let old = transitions.insert(src, Transitions { byte_transitions, ref_transitions });
            // See `if src == accept { ... }` above. The comment there explains
            // why this assert is valid.
            assert_eq!(old, None);
        }

        Self { transitions, start, accept }
    }

    pub(crate) fn get_uninit_edge_dst(&self, state: State) -> Option<State> {
        let transitions = self.transitions.get(&state)?;
        transitions.byte_transitions.get_uninit_edge_dst()
    }

    pub(crate) fn bytes_from(&self, start: State) -> impl Iterator<Item = (Byte, State)> {
        self.transitions
            .get(&start)
            .into_iter()
            .flat_map(|transitions| transitions.byte_transitions.iter())
    }

    pub(crate) fn refs_from(&self, start: State) -> impl Iterator<Item = (Reference<R, T>, State)> {
        self.transitions
            .get(&start)
            .into_iter()
            .flat_map(|transitions| transitions.ref_transitions.iter())
            .map(|(r, s)| (*r, *s))
    }

    #[cfg(test)]
    pub(crate) fn from_edges<B: Clone + Into<Byte>>(
        start: u32,
        accept: u32,
        edges: &[(u32, B, u32)],
    ) -> Self {
        let start = State(start);
        let accept = State(accept);
        let mut transitions: Map<State, Vec<(Byte, State)>> = Map::default();

        for &(src, ref edge, dst) in edges.iter() {
            transitions.entry(State(src)).or_default().push((edge.clone().into(), State(dst)));
        }

        let transitions = transitions
            .into_iter()
            .map(|(src, edges)| {
                (
                    src,
                    Transitions {
                        byte_transitions: EdgeSet::from_edges(edges),
                        ref_transitions: Map::default(),
                    },
                )
            })
            .collect();

        Self { start, accept, transitions }
    }
}

/// Serialize the DFA using the Graphviz DOT format.
impl<R, T> fmt::Debug for Dfa<R, T>
where
    R: Region,
    T: Type,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "digraph {{")?;
        writeln!(f, "    {:?} [shape = doublecircle]", self.start)?;
        writeln!(f, "    {:?} [shape = doublecircle]", self.accept)?;

        for (src, transitions) in self.transitions.iter() {
            for (t, dst) in transitions.byte_transitions.iter() {
                writeln!(f, "    {src:?} -> {dst:?} [label=\"{t:?}\"]")?;
            }

            for (t, dst) in transitions.ref_transitions.iter() {
                writeln!(f, "    {src:?} -> {dst:?} [label=\"{t:?}\"]")?;
            }
        }

        writeln!(f, "}}")
    }
}

use edge_set::EdgeSet;
mod edge_set {
    use smallvec::SmallVec;

    use super::*;

    /// The set of outbound byte edges associated with a DFA node.
    #[derive(Eq, PartialEq, Clone, Debug)]
    pub(super) struct EdgeSet<S = State> {
        // A sequence of byte edges with contiguous byte values and a common
        // destination is stored as a single run.
        //
        // Runs are non-empty, non-overlapping, and stored in ascending order.
        runs: SmallVec<[(Byte, S); 1]>,
    }

    impl<S> EdgeSet<S> {
        pub(crate) fn new(range: Byte, dst: S) -> Self {
            let mut this = Self { runs: SmallVec::new() };
            if !range.is_empty() {
                this.runs.push((range, dst));
            }
            this
        }

        pub(crate) fn empty() -> Self {
            Self { runs: SmallVec::new() }
        }

        #[cfg(test)]
        pub(crate) fn from_edges(mut edges: Vec<(Byte, S)>) -> Self
        where
            S: Ord,
        {
            edges.sort();
            Self { runs: edges.into() }
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = (Byte, S)>
        where
            S: Copy,
        {
            self.runs.iter().copied()
        }

        pub(crate) fn get_uninit_edge_dst(&self) -> Option<S>
        where
            S: Copy,
        {
            // Uninit is ordered last.
            let &(range, dst) = self.runs.last()?;
            if range.contains_uninit() { Some(dst) } else { None }
        }

        pub(crate) fn map_states<SS>(self, mut f: impl FnMut(S) -> SS) -> EdgeSet<SS> {
            EdgeSet {
                // NOTE: It appears as through `<Vec<_> as
                // IntoIterator>::IntoIter` and `std::iter::Map` both implement
                // `TrustedLen`, which in turn means that this `.collect()`
                // allocates the correct number of elements once up-front [1].
                //
                // [1] https://doc.rust-lang.org/1.85.0/src/alloc/vec/spec_from_iter_nested.rs.html#47
                runs: self.runs.into_iter().map(|(b, s)| (b, f(s))).collect(),
            }
        }

        /// Unions two edge sets together.
        ///
        /// If `u = a.union(b)`, then for each byte value, `u` will have an edge
        /// with that byte value and with the destination `join(Some(_), None)`,
        /// `join(None, Some(_))`, or `join(Some(_), Some(_))` depending on whether `a`,
        /// `b`, or both have an edge with that byte value.
        ///
        /// If neither `a` nor `b` have an edge with a particular byte value,
        /// then no edge with that value will be present in `u`.
        pub(crate) fn union(
            &self,
            other: &Self,
            mut join: impl FnMut(Option<S>, Option<S>) -> S,
        ) -> EdgeSet<S>
        where
            S: Copy + Eq,
        {
            let mut runs: SmallVec<[(Byte, S); 1]> = SmallVec::new();
            let xs = self.runs.iter().copied();
            let ys = other.runs.iter().copied();
            for (range, (x, y)) in union(xs, ys) {
                let state = join(x, y);
                match runs.last_mut() {
                    // Merge contiguous runs with a common destination.
                    Some(&mut (ref mut last_range, ref mut last_state))
                        if last_range.end == range.start && *last_state == state =>
                    {
                        last_range.end = range.end
                    }
                    _ => runs.push((range, state)),
                }
            }
            EdgeSet { runs }
        }
    }
}

/// Merges two sorted sequences into one sorted sequence.
pub(crate) fn union<S: Copy, X: Iterator<Item = (Byte, S)>, Y: Iterator<Item = (Byte, S)>>(
    xs: X,
    ys: Y,
) -> UnionIter<X, Y> {
    UnionIter { xs: xs.peekable(), ys: ys.peekable() }
}

pub(crate) struct UnionIter<X: Iterator, Y: Iterator> {
    xs: Peekable<X>,
    ys: Peekable<Y>,
}

// FIXME(jswrenn) we'd likely benefit from specializing try_fold here.
impl<S: Copy, X: Iterator<Item = (Byte, S)>, Y: Iterator<Item = (Byte, S)>> Iterator
    for UnionIter<X, Y>
{
    type Item = (Byte, (Option<S>, Option<S>));

    fn next(&mut self) -> Option<Self::Item> {
        use std::cmp::{self, Ordering};

        let ret;
        match (self.xs.peek_mut(), self.ys.peek_mut()) {
            (None, None) => {
                ret = None;
            }
            (Some(x), None) => {
                ret = Some((x.0, (Some(x.1), None)));
                self.xs.next();
            }
            (None, Some(y)) => {
                ret = Some((y.0, (None, Some(y.1))));
                self.ys.next();
            }
            (Some(x), Some(y)) => {
                let start;
                let end;
                let dst;
                match x.0.start.cmp(&y.0.start) {
                    Ordering::Less => {
                        start = x.0.start;
                        end = cmp::min(x.0.end, y.0.start);
                        dst = (Some(x.1), None);
                    }
                    Ordering::Greater => {
                        start = y.0.start;
                        end = cmp::min(x.0.start, y.0.end);
                        dst = (None, Some(y.1));
                    }
                    Ordering::Equal => {
                        start = x.0.start;
                        end = cmp::min(x.0.end, y.0.end);
                        dst = (Some(x.1), Some(y.1));
                    }
                }
                ret = Some((Byte { start, end }, dst));
                if start == x.0.start {
                    x.0.start = end;
                }
                if start == y.0.start {
                    y.0.start = end;
                }
                if x.0.is_empty() {
                    self.xs.next();
                }
                if y.0.is_empty() {
                    self.ys.next();
                }
            }
        }
        ret
    }
}
