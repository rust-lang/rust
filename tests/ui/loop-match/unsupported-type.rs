// Test that the right error is emitted when the `#[loop_match]` state is an
// unsupported type.

#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

fn unsupported_type() {
    let mut state = Some(false);
    #[loop_match]
    'a: loop {
        state = 'blk: {
            //~^ ERROR this `#[loop_match]` state value has type `Option<bool>`, which is not supported
            match state {
                Some(false) => {
                    #[const_continue]
                    break 'blk Some(true);
                }
                Some(true) => {
                    #[const_continue]
                    break 'blk None;
                }
                None => break 'a,
            }
        }
    }
}
