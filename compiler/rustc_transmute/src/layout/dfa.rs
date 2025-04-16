use itertools::Itertools;
use tracing::instrument;

use super::automaton::{Automaton, State, Transition};
use super::{Byte, Nfa, Ref};
use crate::{Map, Set};

#[derive(PartialEq, Clone, Debug)]
pub(crate) struct Dfa<R: Ref>(
    // INVARIANT: `Automaton` is a DFA, which means that, for any `state`, each
    // transition in `self.0.transitions[state]` contains exactly one
    // destination state.
    pub(crate) Automaton<R>,
);

impl<R> Dfa<R>
where
    R: Ref,
{
    #[cfg(test)]
    pub(crate) fn bool() -> Self {
        let mut transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let start = State::new();
        let accept = State::new();

        transitions
            .entry(start)
            .or_default()
            .insert(Transition::Byte(Byte::Init(0x00)), [accept].into_iter().collect());

        transitions
            .entry(start)
            .or_default()
            .insert(Transition::Byte(Byte::Init(0x01)), [accept].into_iter().collect());

        Dfa(Automaton { transitions, start, accept })
    }

    #[instrument(level = "debug")]
    pub(crate) fn from_nfa(nfa: Nfa<R>) -> Self {
        // It might already be the case that `nfa` is a DFA. If that's the case,
        // we can avoid reconstructing the DFA.
        let is_dfa = nfa
            .0
            .transitions
            .iter()
            .flat_map(|(_, transitions)| transitions.iter())
            .all(|(_, dsts)| dsts.len() <= 1);
        if is_dfa {
            return Dfa(nfa.0);
        }

        let Nfa(Automaton { transitions: nfa_transitions, start: nfa_start, accept: nfa_accept }) =
            nfa;

        let mut dfa_transitions: Map<State, Map<Transition<R>, Set<State>>> = Map::default();
        let mut nfa_to_dfa: Map<State, State> = Map::default();
        let dfa_start = State::new();
        nfa_to_dfa.insert(nfa_start, dfa_start);

        let mut queue = vec![(nfa_start, dfa_start)];

        while let Some((nfa_state, dfa_state)) = queue.pop() {
            if nfa_state == nfa_accept {
                continue;
            }

            for (nfa_transition, next_nfa_states) in nfa_transitions[&nfa_state].iter() {
                use itertools::Itertools as _;

                let dfa_transitions =
                    dfa_transitions.entry(dfa_state).or_insert_with(Default::default);

                let mapped_state = next_nfa_states.iter().find_map(|x| nfa_to_dfa.get(x).copied());

                let next_dfa_state = dfa_transitions.entry(*nfa_transition).or_insert_with(|| {
                    [mapped_state.unwrap_or_else(State::new)].into_iter().collect()
                });
                let next_dfa_state = *next_dfa_state.iter().exactly_one().unwrap();

                for &next_nfa_state in next_nfa_states {
                    nfa_to_dfa.entry(next_nfa_state).or_insert_with(|| {
                        queue.push((next_nfa_state, next_dfa_state));
                        next_dfa_state
                    });
                }
            }
        }

        let dfa_accept = nfa_to_dfa[&nfa_accept];
        Dfa(Automaton { transitions: dfa_transitions, start: dfa_start, accept: dfa_accept })
    }

    pub(crate) fn byte_from(&self, start: State, byte: Byte) -> Option<State> {
        Some(
            self.0
                .transitions
                .get(&start)?
                .get(&Transition::Byte(byte))?
                .iter()
                .copied()
                .exactly_one()
                .unwrap(),
        )
    }

    pub(crate) fn iter_bytes_from(&self, start: State) -> impl Iterator<Item = (Byte, State)> {
        self.0.transitions.get(&start).into_iter().flat_map(|transitions| {
            transitions.iter().filter_map(|(t, s)| {
                let s = s.iter().copied().exactly_one().unwrap();
                if let Transition::Byte(b) = t { Some((*b, s)) } else { None }
            })
        })
    }

    pub(crate) fn iter_refs_from(&self, start: State) -> impl Iterator<Item = (R, State)> {
        self.0.transitions.get(&start).into_iter().flat_map(|transitions| {
            transitions.iter().filter_map(|(t, s)| {
                let s = s.iter().copied().exactly_one().unwrap();
                if let Transition::Ref(r) = t { Some((*r, s)) } else { None }
            })
        })
    }
}
