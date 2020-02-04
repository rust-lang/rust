// make sure that TryInto is implemented for NonZero{integer}
//
// check-pass
use std::num::NonZeroUsize;
use std::convert::TryInto;

fn main() {
    let a: NonZeroUsize = 1_usize.try_into().unwrap();
    let b: NonZeroUsize = 0_usize.try_into().unwrap();
}
