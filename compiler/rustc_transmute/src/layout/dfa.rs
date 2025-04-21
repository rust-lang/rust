use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Byte, Ref, Tree, Uninhabited};
use crate::Map;

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
    byte_transitions: Map<Byte, State>,
    ref_transitions: Map<R, State>,
}

impl<R> Default for Transitions<R>
where
    R: Ref,
{
    fn default() -> Self {
        Self { byte_transitions: Map::default(), ref_transitions: Map::default() }
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
        let mut transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accept = State::new();

        transitions.entry(start).or_default().byte_transitions.insert(Byte::Init(0x00), accept);

        transitions.entry(start).or_default().byte_transitions.insert(Byte::Init(0x01), accept);

        Self { transitions, start, accept }
    }

    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accept = start;

        Self { transitions, start, accept }
    }

    pub(crate) fn from_byte(byte: Byte) -> Self {
        let mut transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accept = State::new();

        transitions.entry(start).or_default().byte_transitions.insert(byte, accept);

        Self { transitions, start, accept }
    }

    pub(crate) fn from_ref(r: R) -> Self {
        let mut transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accept = State::new();

        transitions.entry(start).or_default().ref_transitions.insert(r, accept);

        Self { transitions, start, accept }
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
            let entry = transitions.entry(fix_state(source)).or_default();
            for (edge, destination) in transition.byte_transitions {
                entry.byte_transitions.insert(edge, fix_state(destination));
            }
            for (edge, destination) in transition.ref_transitions {
                entry.ref_transitions.insert(edge, fix_state(destination));
            }
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
        let mut queue = vec![(Some(a.start), Some(b.start))];
        let empty_transitions = Transitions::default();

        while let Some((a_src, b_src)) = queue.pop() {
            let a_transitions =
                a_src.and_then(|a_src| a.transitions.get(&a_src)).unwrap_or(&empty_transitions);
            let b_transitions =
                b_src.and_then(|b_src| b.transitions.get(&b_src)).unwrap_or(&empty_transitions);

            let byte_transitions =
                a_transitions.byte_transitions.keys().chain(b_transitions.byte_transitions.keys());

            for byte_transition in byte_transitions {
                let a_dst = a_transitions.byte_transitions.get(byte_transition).copied();
                let b_dst = b_transitions.byte_transitions.get(byte_transition).copied();

                assert!(a_dst.is_some() || b_dst.is_some());

                let src = mapped((a_src, b_src));
                let dst = mapped((a_dst, b_dst));

                transitions.entry(src).or_default().byte_transitions.insert(*byte_transition, dst);

                if !transitions.contains_key(&dst) {
                    queue.push((a_dst, b_dst))
                }
            }

            let ref_transitions =
                a_transitions.ref_transitions.keys().chain(b_transitions.ref_transitions.keys());

            for ref_transition in ref_transitions {
                let a_dst = a_transitions.ref_transitions.get(ref_transition).copied();
                let b_dst = b_transitions.ref_transitions.get(ref_transition).copied();

                assert!(a_dst.is_some() || b_dst.is_some());

                let src = mapped((a_src, b_src));
                let dst = mapped((a_dst, b_dst));

                transitions.entry(src).or_default().ref_transitions.insert(*ref_transition, dst);

                if !transitions.contains_key(&dst) {
                    queue.push((a_dst, b_dst))
                }
            }
        }

        Self { transitions, start, accept }
    }

    pub(crate) fn bytes_from(&self, start: State) -> Option<&Map<Byte, State>> {
        Some(&self.transitions.get(&start)?.byte_transitions)
    }

    pub(crate) fn byte_from(&self, start: State, byte: Byte) -> Option<State> {
        self.transitions.get(&start)?.byte_transitions.get(&byte).copied()
    }

    pub(crate) fn refs_from(&self, start: State) -> Option<&Map<R, State>> {
        Some(&self.transitions.get(&start)?.ref_transitions)
    }

    #[cfg(test)]
    pub(crate) fn from_edges<B: Copy + Into<Byte>>(
        start: u32,
        accept: u32,
        edges: &[(u32, B, u32)],
    ) -> Self {
        let start = State(start);
        let accept = State(accept);
        let mut transitions: Map<State, Transitions<R>> = Map::default();

        for &(src, edge, dst) in edges {
            let src = State(src);
            let dst = State(dst);
            let old = transitions.entry(src).or_default().byte_transitions.insert(edge.into(), dst);
            assert!(old.is_none());
        }

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
