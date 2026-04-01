#![allow(incomplete_features)]
#![feature(loop_match)]

#[derive(Clone, Copy)]
enum State {
    A,
    B,
}

fn main() {
    let mut state = State::A;

    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk State::B;
                }
                State::B => {
                    return;
                }
            }
        }
    }

    || {
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    //~^ ERROR invalid match on `#[loop_match]` state
                    State::A => {
                        #[const_continue]
                        break 'blk State::B;
                    }
                    State::B => {
                        return;
                    }
                }
            }
        }
    };

    || {
        let mut state = state;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    State::A => {
                        #[const_continue]
                        break 'blk State::B;
                    }
                    State::B => {
                        return;
                    }
                }
            }
        }
    };

    move || {
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    //~^ ERROR invalid match on `#[loop_match]` state
                    State::A => {
                        #[const_continue]
                        break 'blk State::B;
                    }
                    State::B => {
                        return;
                    }
                }
            }
        }
    };
}
