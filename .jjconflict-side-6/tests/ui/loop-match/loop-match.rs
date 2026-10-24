// Test that a basic correct example of `#[loop_match]` with `#[const_continue]`
// works correctly.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

enum State {
    A,
    B,
    C,
}

fn main() {
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk State::B;
                }
                State::B => {
                    // Without special logic, the compiler believes this is a
                    // reassignment to an immutable variable because of the
                    // `loop`. So this tests that local variables work.
                    let _a = 0;

                    if true {
                        #[const_continue]
                        break 'blk State::C;
                    } else {
                        #[const_continue]
                        break 'blk State::A;
                    }
                }
                State::C => break 'a,
            }
        };
    }

    assert!(matches!(state, State::C))
}
