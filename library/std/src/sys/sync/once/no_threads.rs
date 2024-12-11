use crate::cell::Cell;
use crate::sync as public;
use crate::sync::once::ExclusiveState;

pub struct Once {
    state: Cell<State>,
}

pub struct OnceState {
    poisoned: bool,
    set_state_to: Cell<State>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Incomplete,
    Poisoned,
    Running,
    Complete,
}

struct CompletionGuard<'a> {
    state: &'a Cell<State>,
    set_state_on_drop_to: State,
}

impl<'a> Drop for CompletionGuard<'a> {
    fn drop(&mut self) {
        self.state.set(self.set_state_on_drop_to);
    }
}

// Safety: threads are not supported on this platform.
unsafe impl Sync for Once {}

impl Once {
    #[inline]
    pub const fn new() -> Once {
        Once { state: Cell::new(State::Incomplete) }
    }

    #[inline]
    pub fn is_completed(&self) -> bool {
        self.state.get() == State::Complete
    }

    #[inline]
    pub(crate) fn state(&mut self) -> ExclusiveState {
        match self.state.get() {
            State::Incomplete => ExclusiveState::Incomplete,
            State::Poisoned => ExclusiveState::Poisoned,
            State::Complete => ExclusiveState::Complete,
            _ => unreachable!("invalid Once state"),
        }
    }

    #[inline]
    pub(crate) fn set_state(&mut self, new_state: ExclusiveState) {
        self.state.set(match new_state {
            ExclusiveState::Incomplete => State::Incomplete,
            ExclusiveState::Poisoned => State::Poisoned,
            ExclusiveState::Complete => State::Complete,
        });
    }

    #[cold]
    #[track_caller]
    pub fn wait(&self, _ignore_poisoning: bool) {
        panic!("not implementable on this target");
    }

    #[cold]
    #[track_caller]
    pub fn call(&self, ignore_poisoning: bool, f: &mut impl FnMut(&public::OnceState)) {
        let state = self.state.get();
        match state {
            State::Poisoned if !ignore_poisoning => {
                // Panic to propagate the poison.
                panic!("Once instance has previously been poisoned");
            }
            State::Incomplete | State::Poisoned => {
                self.state.set(State::Running);
                // `guard` will set the new state on drop.
                let mut guard =
                    CompletionGuard { state: &self.state, set_state_on_drop_to: State::Poisoned };
                // Run the function, letting it know if we're poisoned or not.
                let f_state = public::OnceState {
                    inner: OnceState {
                        poisoned: state == State::Poisoned,
                        set_state_to: Cell::new(State::Complete),
                    },
                };
                f(&f_state);
                guard.set_state_on_drop_to = f_state.inner.set_state_to.get();
            }
            State::Running => {
                panic!("one-time initialization may not be performed recursively");
            }
            State::Complete => {}
        }
    }
}

impl OnceState {
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[inline]
    pub fn poison(&self) {
        self.set_state_to.set(State::Poisoned)
    }
}
