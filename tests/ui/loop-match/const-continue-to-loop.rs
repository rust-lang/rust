// Test that a `#[const_continue]` that breaks to the label of the loop itself
// rather than to the label of the block within the `#[loop_match]` produces an
// error.

#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

fn const_continue_to_loop() -> u8 {
    let mut state = 0;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk 1;
                }
                _ => {
                    #[const_continue]
                    break 'a 2;
                    //~^ ERROR `#[const_continue]` must break to a labeled block that participates in a `#[loop_match]`
                }
            }
        }
    }
}
