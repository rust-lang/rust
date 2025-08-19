// Test that a `break` without `#[const_continue]` still works as expected.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

fn main() {
    assert_eq!(helper(), 1);
}

fn helper() -> u8 {
    let mut state = 0u8;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                0 => break 'blk 1,
                _ => break 'a state,
            }
        }
    }
}
