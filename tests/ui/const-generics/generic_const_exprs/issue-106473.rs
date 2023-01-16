#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const DEFAULT: u32 = 1;

struct V<const U: usize = DEFAULT> //~ ERROR mismatched types
where
    [(); U]:;

trait Tr {}

impl Tr for V {}

fn main() {}
