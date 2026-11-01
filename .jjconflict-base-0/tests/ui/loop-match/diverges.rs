//@ build-pass
//@ compile-flags: -Zvalidate-mir
#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

// Test that a #[loop_match] without an explicit break from the loop generates valid MIR.

fn break_to_block_unit() -> u8 {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                _ => 'b: {
                    break 'b 2;
                }
            }
        }
    }
}

fn break_to_block_value() -> u8 {
    let mut state = 0u8;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                _ => break 'blk state,
            }
        }
    }
}

fn infinite_a(mut state: u8) {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                a => a,
            }
        }
    }
}
