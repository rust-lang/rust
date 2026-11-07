//! Regression test for <https://github.com/rust-lang/rust/issues/18611>.

fn add_state(op: <isize as HasState>::State) {
//~^ ERROR `isize: HasState` is not satisfied
//~| ERROR `isize: HasState` is not satisfied
}

trait HasState {
    type State;
}

fn main() {}
