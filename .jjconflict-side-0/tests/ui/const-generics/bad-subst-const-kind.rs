//@ incremental
#![crate_type = "lib"]

trait Q {
    const ASSOC: usize;
}

impl<const N: u64> Q for [u8; N] {
    //~^ ERROR: the constant `N` is not of type `usize`
    const ASSOC: usize = 1;
}

pub fn test() -> [u8; <[u8; 13] as Q>::ASSOC] {
    //~^ ERROR: the constant `13` is not of type `u64`
    todo!()
}
