#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

fn const_continue_to_block() -> u8 {
    let state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk 1;
                }
                _ => 'b: {
                    #[const_continue]
                    break 'b 2;
                    //~^ ERROR `#[const_continue]` must break to a labeled block that participates in a `#[loop_match]`
                }
            }
        }
    }
}
