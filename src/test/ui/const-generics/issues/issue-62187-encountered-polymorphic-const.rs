// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl<const L: usize> BitLen for [u8; L] {
    const BIT_LEN: usize = 8 * L;
}

fn main() {
    let foo = <[u8; 2]>::BIT_LEN; //~ WARN unused variable
}
