// incremental
#![crate_type = "lib"]

trait Q {
    const ASSOC: usize;
}

impl<const N: u64> Q for [u8; N] {
    //~^ ERROR mismatched types
    const ASSOC: usize = 1;
}

pub fn test() -> [u8; <[u8; 13] as Q>::ASSOC] { todo!() }
