fn add_state(op: <isize as HasState>::State) {
//~^ ERROR `isize: HasState` is not satisfied
}

trait HasState {
    type State;
}

fn main() {}
