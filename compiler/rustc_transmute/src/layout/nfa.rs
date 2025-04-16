use super::automaton::{Automaton, State, Transition};
use super::{Byte, Ref, Tree, Uninhabited};
use crate::{Map, Set};

/// A non-deterministic finite automaton (NFA) that represents the layout of a type.
/// The transmutability of two given types is computed by comparing their `Nfa`s.
#[derive(PartialEq, Debug)]
pub(crate) struct Nfa<R: Ref>(pub(crate) Automaton<R>);

impl<R> Nfa<R>
where
    R: Ref,
{
    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accept = start;

        Nfa(Automaton { transitions, start, accept })
    }

    pub(crate) fn from_byte(byte: Byte) -> Self {
        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accept = State::new();

        let source = transitions.entry(start).or_default();
        let edge = source.entry(Transition::Byte(byte)).or_default();
        edge.insert(accept);

        Nfa(Automaton { transitions, start, accept })
    }

    pub(crate) fn from_ref(r: R) -> Self {
        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accept = State::new();

        let source = transitions.entry(start).or_default();
        let edge = source.entry(Transition::Ref(r)).or_default();
        edge.insert(accept);

        Nfa(Automaton { transitions, start, accept })
    }

    pub(crate) fn from_tree(tree: Tree<!, R>) -> Result<Self, Uninhabited> {
        Ok(match tree {
            Tree::Byte(b) => Self::from_byte(b),
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
        if self.0.start == self.0.accept {
            return other;
        } else if other.0.start == other.0.accept {
            return self;
        }

        let start = self.0.start;
        let accept = other.0.accept;

        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = self.0.transitions;

        for (source, transition) in other.0.transitions {
            let fix_state = |state| if state == other.0.start { self.0.accept } else { state };
            let entry = transitions.entry(fix_state(source)).or_default();
            for (edge, destinations) in transition {
                let entry = entry.entry(edge).or_default();
                for destination in destinations {
                    entry.insert(fix_state(destination));
                }
            }
        }

        Nfa(Automaton { transitions, start, accept })
    }

    /// Compute the union of two `Nfa`s.
    pub(crate) fn union(self, other: Self) -> Self {
        let start = self.0.start;
        let accept = self.0.accept;

        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> =
            self.0.transitions.clone();

        for (&(mut source), transition) in other.0.transitions.iter() {
            // if source is starting state of `other`, replace with starting state of `self`
            if source == other.0.start {
                source = self.0.start;
            }
            let entry = transitions.entry(source).or_default();
            for (edge, destinations) in transition {
                let entry = entry.entry(*edge).or_default();
                for &(mut destination) in destinations {
                    // if dest is accept state of `other`, replace with accept state of `self`
                    if destination == other.0.accept {
                        destination = self.0.accept;
                    }
                    entry.insert(destination);
                }
            }
        }
        Nfa(Automaton { transitions, start, accept })
    }
}
