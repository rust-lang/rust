// Test that #[const_continue] correctly emits cleanup paths for drops.

//@ run-pass
//@ needs-unwind

#![allow(incomplete_features)]
#![feature(loop_match)]

enum State {
    A,
    B,
}

struct ExitOnDrop;

impl Drop for ExitOnDrop {
    fn drop(&mut self) {
        std::process::exit(0);
    }
}

struct DropBomb;

impl Drop for DropBomb {
    fn drop(&mut self) {
        panic!("this must unwind");
    }
}

fn main() {
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    let _exit = ExitOnDrop;
                    let _bomb = DropBomb;

                    #[const_continue]
                    break 'blk State::B;
                }
                State::B => break 'a,
            }
        };
    }

    unreachable!();
}
