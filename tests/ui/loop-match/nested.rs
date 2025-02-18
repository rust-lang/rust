// Test that a nested `#[loop_match]` works as expected, and that e.g. a
// `#[const_continue]` of the inner `#[loop_match]` does not interact with the
// outer `#[loop_match]`.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

enum State1 {
    A,
    B,
    C,
}

enum State2 {
    X,
    Y,
    Z,
}

fn main() {
    assert_eq!(run(), concat!("ab", "xyz", "xyz", "c"))
}

fn run() -> String {
    let mut accum = String::new();

    let mut state1 = State1::A;
    let mut state2 = State2::X;

    let mut first = true;

    #[loop_match]
    'a: loop {
        state1 = 'blk1: {
            match state1 {
                State1::A => {
                    accum.push('a');
                    #[const_continue]
                    break 'blk1 State1::B;
                }
                State1::B => {
                    accum.push('b');
                    #[loop_match]
                    loop {
                        state2 = 'blk2: {
                            match state2 {
                                State2::X => {
                                    accum.push('x');
                                    #[const_continue]
                                    break 'blk2 State2::Y;
                                }
                                State2::Y => {
                                    accum.push('y');
                                    #[const_continue]
                                    break 'blk2 State2::Z;
                                }
                                State2::Z => {
                                    accum.push('z');
                                    if first {
                                        first = false;
                                        #[const_continue]
                                        break 'blk2 State2::X;
                                    } else {
                                        #[const_continue]
                                        break 'blk1 State1::C;
                                    }
                                }
                            }
                        }
                    }
                }
                State1::C => {
                    accum.push('c');
                    break 'a;
                }
            }
        }
    }

    accum
}
