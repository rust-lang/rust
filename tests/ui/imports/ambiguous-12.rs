// https://github.com/rust-lang/rust/pull/113099#issuecomment-1637022296

macro_rules! m {
    () => {
        pub fn b() {}
    };
}

pub mod ciphertext {
    m!();
}
pub mod public {
    use crate::ciphertext::*;
    m!();
}

use crate::ciphertext::*;
use crate::public::*;

fn main() {
    b();
    //~^ ERROR `b` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
