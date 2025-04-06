// Test that `#[loop_match]` and `#[const_continue]` cannot be used without
// `#![feature(loop_match)]`.

enum State {
    A,
    B,
    C,
}

fn main() {
    let mut state = State::A;
    #[loop_match] //~ ERROR the `#[loop_match]` attribute is an experimental feature
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    //~^ ERROR the `#[const_continue]` attribute is an experimental feature
                    break 'blk State::B;
                }
                State::B => {
                    #[const_continue]
                    //~^ ERROR the `#[const_continue]` attribute is an experimental feature
                    break 'blk State::C;
                }
                State::C => break 'a,
            }
        };
    }
}
