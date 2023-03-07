use super::{nfa, Byte, Nfa, Ref};
use crate::Map;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(PartialEq, Clone, Debug)]
pub(crate) struct Dfa<R>
where
    R: Ref,
{
    pub(crate) transitions: Map<State, Transitions<R>>,
    pub(crate) start: State,
    pub(crate) accepting: State,
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

impl<R> Transitions<R>
where
    R: Ref,
{
    fn insert(&mut self, transition: Transition<R>, state: State) {
        match transition {
            Transition::Byte(b) => {
                self.byte_transitions.insert(b, state);
            }
            Transition::Ref(r) => {
                self.ref_transitions.insert(r, state);
            }
        }
    }
}

/// The states in a `Nfa` represent byte offsets.
#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Copy, Clone)]
pub(crate) struct State(u32);

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

impl<R> Dfa<R>
where
    R: Ref,
{
    pub(crate) fn unit() -> Self {
        let transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accepting = start;

        Self { transitions, start, accepting }
    }

    #[cfg(test)]
    pub(crate) fn bool() -> Self {
        let mut transitions: Map<State, Transitions<R>> = Map::default();
        let start = State::new();
        let accepting = State::new();

        transitions.entry(start).or_default().insert(Transition::Byte(Byte::Init(0x00)), accepting);

        transitions.entry(start).or_default().insert(Transition::Byte(Byte::Init(0x01)), accepting);

        Self { transitions, start, accepting }
    }

    #[instrument(level = "debug")]
    pub(crate) fn from_nfa(nfa: Nfa<R>) -> Self {
        let Nfa { transitions: nfa_transitions, start: nfa_start, accepting: nfa_accepting } = nfa;

        let mut dfa_transitions: Map<State, Transitions<R>> = Map::default();
        let mut nfa_to_dfa: Map<nfa::State, State> = Map::default();
        let dfa_start = State::new();
        nfa_to_dfa.insert(nfa_start, dfa_start);

        let mut queue = vec![(nfa_start, dfa_start)];

        while let Some((nfa_state, dfa_state)) = queue.pop() {
            if nfa_state == nfa_accepting {
                continue;
            }

            for (nfa_transition, next_nfa_states) in nfa_transitions[&nfa_state].iter() {
                let dfa_transitions =
                    dfa_transitions.entry(dfa_state).or_insert_with(Default::default);

                let mapped_state = next_nfa_states.iter().find_map(|x| nfa_to_dfa.get(x).copied());

                let next_dfa_state = match nfa_transition {
                    &nfa::Transition::Byte(b) => *dfa_transitions
                        .byte_transitions
                        .entry(b)
                        .or_insert_with(|| mapped_state.unwrap_or_else(State::new)),
                    &nfa::Transition::Ref(r) => *dfa_transitions
                        .ref_transitions
                        .entry(r)
                        .or_insert_with(|| mapped_state.unwrap_or_else(State::new)),
                };

                for &next_nfa_state in next_nfa_states {
                    nfa_to_dfa.entry(next_nfa_state).or_insert_with(|| {
                        queue.push((next_nfa_state, next_dfa_state));
                        next_dfa_state
                    });
                }
            }
        }

        let dfa_accepting = nfa_to_dfa[&nfa_accepting];

        Self { transitions: dfa_transitions, start: dfa_start, accepting: dfa_accepting }
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
}

impl State {
    pub(crate) fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl<R> From<nfa::Transition<R>> for Transition<R>
where
    R: Ref,
{
    fn from(nfa_transition: nfa::Transition<R>) -> Self {
        match nfa_transition {
            nfa::Transition::Byte(byte) => Transition::Byte(byte),
            nfa::Transition::Ref(r) => Transition::Ref(r),
        }
    }
}
