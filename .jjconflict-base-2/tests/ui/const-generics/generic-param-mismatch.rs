fn test<const N: usize, const M: usize>() -> [u8; M] {
    [0; N] //~ ERROR mismatched types
}

fn main() {}
