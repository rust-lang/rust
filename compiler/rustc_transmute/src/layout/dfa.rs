use std::fmt;
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Byte, Ref, Tree, Uninhabited};
use crate::{Map, Set};

#[derive(PartialEq)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct Dfa<R>
where
    R: Ref,
{
    pub(crate) transitions: Map<State, Transitions<R>>,
    pub(crate) start: State,
    pub(crate) accept: State,
}

#[derive(PartialEq, Clone, Debug)]
pub(crate) struct Transitions<R>
where
    R: Ref,
{
    byte_transitions: EdgeSet<State>,
    ref_transitions: Map<R, State>,
}

impl<R> Default for Transitions<R>
where
    R: Ref,
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

impl<R> Dfa<R>
where
    R: Ref,
{
    #[cfg(test)]
    pub(crate) fn bool() -> Self {
        Self::from_transitions(|accept| Transitions {
            byte_transitions: EdgeSet::new(Byte::new(0x00..=0x01), accept),
            ref_transitions: Map::default(),
        })
    }

    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Transitions<R>> = Map::default();
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

    pub(crate) fn from_ref(r: R) -> Self {
        Self::from_transitions(|accept| Transitions {
            byte_transitions: EdgeSet::empty(),
            ref_transitions: [(r, accept)].into_iter().collect(),
        })
    }

    fn from_transitions(f: impl FnOnce(State) -> Transitions<R>) -> Self {
        let start = State::new();
        let accept = State::new();

        Self { transitions: [(start, f(accept))].into_iter().collect(), start, accept }
    }

    pub(crate) fn from_tree(tree: Tree<!, R>) -> Result<Self, Uninhabited> {
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

        let mut transitions: Map<State, Transitions<R>> = self.transitions;

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
        let mut transitions: Map<State, Transitions<R>> = Map::default();
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

            let byte_transitions =
                a_transitions.byte_transitions.union(&b_transitions.byte_transitions);

            let byte_transitions = byte_transitions.map_states(|(a_dst, b_dst)| {
                assert!(a_dst.is_some() || b_dst.is_some());

                queue.enqueue(a_dst, b_dst);
                mapped((a_dst, b_dst))
            });

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

    pub(crate) fn states_from(
        &self,
        state: State,
        src_validity: RangeInclusive<u8>,
    ) -> impl Iterator<Item = (Byte, State)> {
        self.transitions
            .get(&state)
            .map(move |t| t.byte_transitions.states_from(src_validity))
            .into_iter()
            .flatten()
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

    pub(crate) fn refs_from(&self, start: State) -> impl Iterator<Item = (R, State)> {
        self.transitions
            .get(&start)
            .into_iter()
            .flat_map(|transitions| transitions.ref_transitions.iter())
            .map(|(r, s)| (*r, *s))
    }

    #[cfg(test)]
    pub(crate) fn from_edges<B: Copy + Into<Byte>>(
        start: u32,
        accept: u32,
        edges: &[(u32, B, u32)],
    ) -> Self {
        let start = State(start);
        let accept = State(accept);
        let mut transitions: Map<State, Vec<(Byte, State)>> = Map::default();

        for (src, edge, dst) in edges.iter().copied() {
            transitions.entry(State(src)).or_default().push((edge.into(), State(dst)));
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
impl<R> fmt::Debug for Dfa<R>
where
    R: Ref,
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
    use std::cmp;

    use run::*;
    use smallvec::{SmallVec, smallvec};

    use super::*;
    mod run {
        use std::ops::{Range, RangeInclusive};

        use super::*;
        use crate::layout::Byte;

        /// A logical set of edges.
        ///
        /// A `Run` encodes one edge for every byte value in `start..=end`
        /// pointing to `dst`.
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub(super) struct Run<S> {
            // `start` and `end` are both inclusive (ie, closed) bounds, as this
            // is required in order to be able to store 0..=255. We provide
            // setters and getters which operate on closed/open ranges, which
            // are more intuitive and easier for performing offset math.
            start: u8,
            end: u8,
            pub(super) dst: S,
        }

        impl<S> Run<S> {
            pub(super) fn new(range: RangeInclusive<u8>, dst: S) -> Self {
                Self { start: *range.start(), end: *range.end(), dst }
            }

            pub(super) fn from_inclusive_exclusive(range: Range<u16>, dst: S) -> Self {
                Self {
                    start: range.start.try_into().unwrap(),
                    end: (range.end - 1).try_into().unwrap(),
                    dst,
                }
            }

            pub(super) fn contains(&self, idx: u16) -> bool {
                idx >= u16::from(self.start) && idx <= u16::from(self.end)
            }

            pub(super) fn as_inclusive_exclusive(&self) -> (u16, u16) {
                (u16::from(self.start), u16::from(self.end) + 1)
            }

            pub(super) fn as_byte(&self) -> Byte {
                Byte::new(self.start..=self.end)
            }

            pub(super) fn map_state<SS>(self, f: impl FnOnce(S) -> SS) -> Run<SS> {
                let Run { start, end, dst } = self;
                Run { start, end, dst: f(dst) }
            }

            /// Produces a new `Run` whose lower bound is the greater of
            /// `self`'s existing lower bound and `lower_bound`.
            pub(super) fn clamp_lower(self, lower_bound: u8) -> Self {
                let Run { start, end, dst } = self;
                Run { start: cmp::max(start, lower_bound), end, dst }
            }
        }
    }

    /// The set of outbound byte edges associated with a DFA node (not including
    /// reference edges).
    #[derive(Eq, PartialEq, Clone, Debug)]
    pub(super) struct EdgeSet<S = State> {
        // A sequence of runs stored in ascending order. Since the graph is a
        // DFA, these must be non-overlapping with one another.
        runs: SmallVec<[Run<S>; 1]>,
        // The edge labeled with the uninit byte, if any.
        //
        // FIXME(@joshlf): Make `State` a `NonZero` so that this is NPO'd.
        uninit: Option<S>,
    }

    impl<S> EdgeSet<S> {
        pub(crate) fn new(byte: Byte, dst: S) -> Self {
            match byte.range() {
                Some(range) => Self { runs: smallvec![Run::new(range, dst)], uninit: None },
                None => Self { runs: SmallVec::new(), uninit: Some(dst) },
            }
        }

        pub(crate) fn empty() -> Self {
            Self { runs: SmallVec::new(), uninit: None }
        }

        #[cfg(test)]
        pub(crate) fn from_edges(mut edges: Vec<(Byte, S)>) -> Self
        where
            S: Ord,
        {
            edges.sort();
            Self {
                runs: edges
                    .into_iter()
                    .map(|(byte, state)| Run::new(byte.range().unwrap(), state))
                    .collect(),
                uninit: None,
            }
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = (Byte, S)>
        where
            S: Copy,
        {
            self.uninit
                .map(|dst| (Byte::uninit(), dst))
                .into_iter()
                .chain(self.runs.iter().map(|run| (run.as_byte(), run.dst)))
        }

        pub(crate) fn states_from(
            &self,
            byte: RangeInclusive<u8>,
        ) -> impl Iterator<Item = (Byte, S)>
        where
            S: Copy,
        {
            // FIXME(@joshlf): Optimize this. A manual scan over `self.runs` may
            // permit us to more efficiently discard runs which will not be
            // produced by this iterator.
            self.iter().filter(move |(o, _)| Byte::new(byte.clone()).transmutable_into(&o))
        }

        pub(crate) fn get_uninit_edge_dst(&self) -> Option<S>
        where
            S: Copy,
        {
            self.uninit
        }

        pub(crate) fn map_states<SS>(self, mut f: impl FnMut(S) -> SS) -> EdgeSet<SS> {
            EdgeSet {
                // NOTE: It appears as through `<Vec<_> as
                // IntoIterator>::IntoIter` and `std::iter::Map` both implement
                // `TrustedLen`, which in turn means that this `.collect()`
                // allocates the correct number of elements once up-front [1].
                //
                // [1] https://doc.rust-lang.org/1.85.0/src/alloc/vec/spec_from_iter_nested.rs.html#47
                runs: self.runs.into_iter().map(|run| run.map_state(&mut f)).collect(),
                uninit: self.uninit.map(f),
            }
        }

        /// Unions two edge sets together.
        ///
        /// If `u = a.union(b)`, then for each byte value, `u` will have an edge
        /// with that byte value and with the destination `(Some(_), None)`,
        /// `(None, Some(_))`, or `(Some(_), Some(_))` depending on whether `a`,
        /// `b`, or both have an edge with that byte value.
        ///
        /// If neither `a` nor `b` have an edge with a particular byte value,
        /// then no edge with that value will be present in `u`.
        pub(crate) fn union(&self, other: &Self) -> EdgeSet<(Option<S>, Option<S>)>
        where
            S: Copy,
        {
            let uninit = match (self.uninit, other.uninit) {
                (None, None) => None,
                (s, o) => Some((s, o)),
            };

            let mut runs = SmallVec::new();

            // Iterate over `self.runs` and `other.runs` simultaneously,
            // advancing `idx` as we go. At each step, we advance `idx` as far
            // as we can without crossing a run boundary in either `self.runs`
            // or `other.runs`.

            // INVARIANT: `idx < s[0].end && idx < o[0].end`.
            let (mut s, mut o) = (self.runs.as_slice(), other.runs.as_slice());
            let mut idx = 0u16;
            while let (Some((s_run, s_rest)), Some((o_run, o_rest))) =
                (s.split_first(), o.split_first())
            {
                let (s_start, s_end) = s_run.as_inclusive_exclusive();
                let (o_start, o_end) = o_run.as_inclusive_exclusive();

                // Compute `end` as the end of the current run (which starts
                // with `idx`).
                let (end, dst) = match (s_run.contains(idx), o_run.contains(idx)) {
                    // `idx` is in an existing run in both `s` and `o`, so `end`
                    // is equal to the smallest of the two ends of those runs.
                    (true, true) => (cmp::min(s_end, o_end), (Some(s_run.dst), Some(o_run.dst))),
                    // `idx` is in an existing run in `s`, but not in any run in
                    // `o`. `end` is either the end of the `s` run or the
                    // beginning of the next `o` run, whichever comes first.
                    (true, false) => (cmp::min(s_end, o_start), (Some(s_run.dst), None)),
                    // The inverse of the previous case.
                    (false, true) => (cmp::min(s_start, o_end), (None, Some(o_run.dst))),
                    // `idx` is not in a run in either `s` or `o`, so advance it
                    // to the beginning of the next run.
                    (false, false) => {
                        idx = cmp::min(s_start, o_start);
                        continue;
                    }
                };

                // FIXME(@joshlf): If this is contiguous with the previous run
                // and has the same `dst`, just merge it into that run rather
                // than adding a new one.
                runs.push(Run::from_inclusive_exclusive(idx..end, dst));
                idx = end;

                if idx >= s_end {
                    s = s_rest;
                }
                if idx >= o_end {
                    o = o_rest;
                }
            }

            // At this point, either `s` or `o` have been exhausted, so the
            // remaining elements in the other slice are guaranteed to be
            // non-overlapping. We can add all remaining runs to `runs` with no
            // further processing.
            if let Ok(idx) = u8::try_from(idx) {
                let (slc, map) = if !s.is_empty() {
                    let map: fn(_) -> _ = |st| (Some(st), None);
                    (s, map)
                } else {
                    let map: fn(_) -> _ = |st| (None, Some(st));
                    (o, map)
                };
                runs.extend(slc.iter().map(|run| run.clamp_lower(idx).map_state(map)));
            }

            EdgeSet { runs, uninit }
        }
    }
}
