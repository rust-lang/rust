#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

const CONST_THAT_PANICS: u8 = panic!("diverge!");
//~^ ERROR: evaluation panicked: diverge!

fn test(mut state: u8) {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk CONST_THAT_PANICS;
                }

                _ => unreachable!(),
            }
        }
    }
}
