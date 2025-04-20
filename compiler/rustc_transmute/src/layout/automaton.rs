use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Byte, Ref};
use crate::{Map, Set};

#[derive(PartialEq, Debug, Clone)]
pub(crate) struct Automaton<R>
where
    R: Ref,
{
    pub(crate) transitions: Map<State, Map<Transition<R>, Set<State>>>,
    pub(crate) start: State,
    pub(crate) accept: State,
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

impl State {
    pub(crate) fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}
