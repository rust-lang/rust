// Test that macros can be defined in the labeled block. This should not trigger an error about
// statements not being allowed in that position, and should of course work as expected.

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
            macro_rules! const_continue {
                ($e:expr) => {
                    #[const_continue]
                    break 'blk $e;
                };
            }
            match state {
                State::A => {
                    const_continue!(State::B);
                }
                State::B => {
                    // Without special logic, the compiler believes this is a
                    // reassignment to an immutable variable because of the
                    // `loop`. So this tests that local variables work.
                    let _a = 0;

                    if true {
                        const_continue!(State::C);
                    } else {
                        const_continue!(State::A);
                    }
                }
                State::C => break 'a,
            }
        };
    }

    assert!(matches!(state, State::C))
}
