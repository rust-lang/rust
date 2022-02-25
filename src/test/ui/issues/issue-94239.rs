pub const fn test_match_range(len: usize) -> usize {
    match len {
        10000000000000000000..=99999999999999999999 => 0, //~ ERROR literal out of range for `usize`
        _ => unreachable!(),
    }
}

fn main() {}
