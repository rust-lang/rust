// run-pass
#![feature(slice_patterns)]

fn main() {
    assert_eq!(match [0u8; 1024] {
        _ => 42_usize,
    }, 42_usize);

    assert_eq!(match [0u8; 1024] {
        [1, _..] => 0_usize,
        [0, _..] => 1_usize,
        _ => 2_usize
    }, 1_usize);
}
