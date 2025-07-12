#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

const fn const_fn() -> i32 {
    1
}

#[unsafe(no_mangle)]
fn suggest_const_block<const N: i32>() -> i32 {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk const_fn();
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                1 => {
                    #[const_continue]
                    break 'blk const { const_fn() };
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                2 => {
                    #[const_continue]
                    break 'blk N;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                _ => {
                    #[const_continue]
                    break 'blk 1 + 1;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
            }
        }
    }
    state
}
