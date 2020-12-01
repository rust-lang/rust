// revisions: full min
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

fn test<const N: usize, const M: usize>() -> [u8; M] {
    [0; N] //~ ERROR mismatched types
}

fn main() {}
