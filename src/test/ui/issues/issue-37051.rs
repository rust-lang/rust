// check-pass

#![feature(associated_type_defaults)]

trait State: Sized {
    type NextState: State = StateMachineEnded;
    fn execute(self) -> Option<Self::NextState>;
}

struct StateMachineEnded;

impl State for StateMachineEnded {
    fn execute(self) -> Option<Self::NextState> {
        None
    }
}

fn main() {}
