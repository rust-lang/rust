#![feature(rustc_attrs, associated_type_defaults)]
#![allow(warnings)]

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

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
