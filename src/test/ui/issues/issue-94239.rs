pub const fn test_match_range(len: u64) -> u64 {
    match len {
        10000000000000000000..=99999999999999999999 => 0, //~ ERROR literal out of range for `u64`
        _ => unreachable!(),
    }
}

fn main() {}
