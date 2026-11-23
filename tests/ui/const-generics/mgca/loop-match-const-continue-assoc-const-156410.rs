//! Regression test for <https://github.com/rust-lang/rust/issues/156410>.
#![feature(gca_min_const_items, loop_match)]

trait T {
    #[rustc_always_gca]
    const N: usize;

    fn a() {
        let mut state;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::N;
                        //~^ ERROR could not determine the target branch for this `#[const_continue]`
                    }
                    _ => todo!(),
                }
            }
        }
    }
}

fn main() {}
