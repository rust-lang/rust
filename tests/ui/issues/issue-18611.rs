fn add_state(op: <isize as HasState>::State) {
//~^ ERROR trait `HasState` is not implemented for `isize`
//~| ERROR trait `HasState` is not implemented for `isize`
}

trait HasState {
    type State;
}

fn main() {}
