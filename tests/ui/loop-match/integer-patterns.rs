// Test that signed and unsigned integer patterns work with #[loop_match].

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

fn main() {
    let mut state = 0i32;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                -1 => {
                    #[const_continue]
                    break 'blk 2;
                }
                0 => {
                    #[const_continue]
                    break 'blk -1;
                }
                2 => break 'a,
                _ => unreachable!("weird value {:?}", state),
            }
        }
    }
    assert_eq!(state, 2);
}
