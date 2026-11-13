//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree

#![allow(incomplete_features)]
#![feature(loop_match)]

fn boolean(mut state: bool) -> bool {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                true => {
                    #[const_continue]
                    break 'blk false;
                }
                false => return state,
            }
        }
    }
}

fn main() {}
