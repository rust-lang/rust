// Test that `#[loop_match]` supports or-patterns.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    A,
    B,
    C,
    D,
}

fn main() {
    let mut states = vec![];
    let mut first = true;
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    states.push(state);
                    if first {
                        #[const_continue]
                        break 'blk State::B;
                    } else {
                        #[const_continue]
                        break 'blk State::D;
                    }
                }
                State::B | State::D => {
                    states.push(state);
                    if first {
                        first = false;
                        #[const_continue]
                        break 'blk State::A;
                    } else {
                        #[const_continue]
                        break 'blk State::C;
                    }
                }
                State::C => {
                    states.push(state);
                    break 'a;
                }
            }
        }
    }

    assert_eq!(states, [State::A, State::B, State::A, State::D, State::C]);
}
