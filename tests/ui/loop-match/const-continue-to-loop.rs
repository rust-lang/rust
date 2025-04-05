#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

fn const_continue_to_loop() -> u8 {
    let state = 0;
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
