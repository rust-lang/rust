// run-pass

// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl<const L: usize> BitLen for [u8; L] {
    const BIT_LEN: usize = 8 * L;
}

fn main() {
    let _foo = <[u8; 2]>::BIT_LEN;
}
