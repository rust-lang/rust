//ignore-windows: Causes a stack overflow?!? Likely a rustc bug: https://github.com/rust-lang/rust/issues/53820
//FIXME: Once that bug is fixed, increase the size to 16*1024 and enable on all platforms.

#![feature(slice_patterns)]

fn main() {
    assert_eq!(match [0u8; 1024] {
        _ => 42_usize,
    }, 42_usize);

    assert_eq!(match [0u8; 1024] {
        [1, ..] => 0_usize,
        [0, ..] => 1_usize,
        _ => 2_usize
    }, 1_usize);
}
