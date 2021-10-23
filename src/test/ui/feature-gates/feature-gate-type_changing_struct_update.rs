#[derive(Debug)]
struct Machine<S> {
    state: S,
    common_field1: &'static str,
    common_field2: i32,
}
#[derive(Debug)]
struct State1;
#[derive(Debug, PartialEq)]
struct State2;

fn update_to_state2() {
    let m1: Machine<State1> = Machine {
        state: State1,
        common_field1: "hello",
        common_field2: 2,
    };
    let m2: Machine<State2> = Machine {
        state: State2,
        ..m1 //~ ERROR mismatched types
    };
    // FIXME: this should trigger feature gate
    assert_eq!(State2, m2.state);
}

fn main() {}
