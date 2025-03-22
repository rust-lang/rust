// Test that signed and unsigned integer patterns work with #[loop_match].

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

fn main() {
    let mut state = 0;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                -1 => {
                    if true {
                        #[const_continue]
                        break 'blk 2;
                    } else {
                        #[const_continue]
                        break 'blk 0;
                    }
                }
                0 => {
                    #[const_continue]
                    break 'blk -1;
                }
                2 => break 'a,
                _ => break 'a,
            }
        }
    }
    assert_eq!(state, 2);
}
