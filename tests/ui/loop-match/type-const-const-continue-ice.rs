// Regression test for <https://github.com/rust-lang/rust/issues/156410>

#![feature(min_generic_const_args)]
#![feature(loop_match)]

trait T {
    type const N: usize;
    fn a() {
        let mut s;
        #[loop_match]
        loop {
            s = 'b: {
                match s {
                    _ => {
                        #[const_continue]
                        break 'b Self::N
                        //~^ ERROR could not determine the target branch for this `#[const_continue]`
                    }
                }
            }
        }
    }
}

fn main() {}
