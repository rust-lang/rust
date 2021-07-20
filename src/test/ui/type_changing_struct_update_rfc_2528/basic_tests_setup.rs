// gate-test-type_changing_struct_update

/// Here, we're documenting the current state of affairs related to
/// FRU (functional record update)

#![feature(type_changing_struct_update)]

struct Machine<S> {
    state: S,
    common_field1: &'static str,
    common_field2: i32,
}

struct State1;
struct State2;

impl Machine<State1> {
    fn into_state2(self) -> Machine<State2> {
        // do stuff
        Machine {
            state: State2,
            ..self
        }
    }
}

#[test]
#[should_panic]  // NOTE: this would be fixed by the implementation of the RFC
fn update_to_state2() {
    let m1: Machine<State1> = Machine {
        state: State1,
        common_field1: "hello",
        common_field2: 2,
    };
    let m2: Machine<State2> = m1.into_state2();
    assert_eq!(State2, m2.state);
}

fn main() {}
