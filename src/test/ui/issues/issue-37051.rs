// compile-pass
// skip-codegen
#![feature(associated_type_defaults)]
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


fn main() {
}
