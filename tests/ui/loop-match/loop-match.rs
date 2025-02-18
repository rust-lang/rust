//@ run-pass

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
}
