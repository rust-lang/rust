use super::{Byte, Ref, Tree, Uninhabited};
use crate::{Map, Set};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// A non-deterministic finite automaton (NFA) that represents the layout of a type.
/// The transmutability of two given types is computed by comparing their `Nfa`s.
#[derive(PartialEq, Debug)]
pub(crate) struct Nfa<R>
where
    R: Ref,
{
    pub(crate) transitions: Map<State, Map<Transition<R>, Set<State>>>,
    pub(crate) start: State,
    pub(crate) accepting: State,
}

/// The states in a `Nfa` represent byte offsets.
#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Copy, Clone)]
pub(crate) struct State(u32);

/// The transitions between states in a `Nfa` reflect bit validity.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub(crate) enum Transition<R>
where
    R: Ref,
{
    Byte(Byte),
    Ref(R),
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S_{}", self.0)
    }
}

impl<R> fmt::Debug for Transition<R>
where
    R: Ref,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Self::Byte(b) => b.fmt(f),
            Self::Ref(r) => r.fmt(f),
        }
    }
}

impl<R> Nfa<R>
where
    R: Ref,
{
    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accepting = start;

        Nfa { transitions, start, accepting }
    }

    pub(crate) fn from_byte(byte: Byte) -> Self {
        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accepting = State::new();

        let source = transitions.entry(start).or_default();
        let edge = source.entry(Transition::Byte(byte)).or_default();
        edge.insert(accepting);

        Nfa { transitions, start, accepting }
    }

    pub(crate) fn from_ref(r: R) -> Self {
        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accepting = State::new();

        let source = transitions.entry(start).or_default();
        let edge = source.entry(Transition::Ref(r)).or_default();
        edge.insert(accepting);

        Nfa { transitions, start, accepting }
    }

    pub(crate) fn from_tree(tree: Tree<!, R>) -> Result<Self, Uninhabited> {
        Ok(match tree {
            Tree::Byte(b) => Self::from_byte(b),
            Tree::Def(..) => unreachable!(),
            Tree::Ref(r) => Self::from_ref(r),
            Tree::Alt(alts) => {
                let mut alts = alts.into_iter().map(Self::from_tree);
                let mut nfa = alts.next().ok_or(Uninhabited)??;
                for alt in alts {
                    nfa = nfa.union(alt?);
                }
                nfa
            }
            Tree::Seq(elts) => {
                let mut nfa = Self::unit();
                for elt in elts.into_iter().map(Self::from_tree) {
                    nfa = nfa.concat(elt?);
                }
                nfa
            }
        })
    }

    /// Concatenate two `Nfa`s.
    pub(crate) fn concat(self, other: Self) -> Self {
        if self.start == self.accepting {
            return other;
        } else if other.start == other.accepting {
            return self;
        }

        let start = self.start;
        let accepting = other.accepting;

        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = self.transitions;

        for (source, transition) in other.transitions {
            let fix_state = |state| if state == other.start { self.accepting } else { state };
            let entry = transitions.entry(fix_state(source)).or_default();
            for (edge, destinations) in transition {
                let entry = entry.entry(edge.clone()).or_default();
                for destination in destinations {
                    entry.insert(fix_state(destination));
                }
            }
        }

        Self { transitions, start, accepting }
    }

    /// Compute the union of two `Nfa`s.
    pub(crate) fn union(self, other: Self) -> Self {
        let start = self.start;
        let accepting = self.accepting;

        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = self.transitions.clone();

        for (&(mut source), transition) in other.transitions.iter() {
            // if source is starting state of `other`, replace with starting state of `self`
            if source == other.start {
                source = self.start;
            }
            let entry = transitions.entry(source).or_default();
            for (edge, destinations) in transition {
                let entry = entry.entry(edge.clone()).or_default();
                for &(mut destination) in destinations {
                    // if dest is accepting state of `other`, replace with accepting state of `self`
                    if destination == other.accepting {
                        destination = self.accepting;
                    }
                    entry.insert(destination);
                }
            }
        }
        Self { transitions, start, accepting }
    }

    pub(crate) fn edges_from(&self, start: State) -> Option<&Map<Transition<R>, Set<State>>> {
        self.transitions.get(&start)
    }
}

impl State {
    pub(crate) fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}
