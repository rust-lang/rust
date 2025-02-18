//@ run-pass

#![feature(loop_match)]

enum State {
    A,
    B,
    C,
    D,
}

fn main() {
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    if true {
                        #[const_continue]
                        break 'blk State::B;
                    } else {
                        #[const_continue]
                        break 'blk State::D;
                    }
                }
                State::B | State::D => {
                    if true {
                        #[const_continue]
                        break 'blk State::C;
                    } else {
                        // No drops allowed at this point
                        #[const_continue]
                        break 'blk State::A;
                    }
                }
                State::C => break 'a,
            }
        }
    }
}
