//! This is a regression test for https://github.com/rust-lang/rust/issues/140686.
//! Although this is a ld64(ld-classic) bug, we still need to support it
//! due to cross-compilation and support for older Xcode.

//@ compile-flags: -Copt-level=3 -Ccodegen-units=256 -Clink-arg=-ld_classic
//@ run-pass
//@ only-x86_64-apple-darwin

fn main() {
    let dst: Vec<u8> = Vec::new();
    let len = broken_func(std::hint::black_box(2), dst);
    assert_eq!(len, 8);
}

#[inline(never)]
pub fn broken_func(version: usize, mut dst: Vec<u8>) -> usize {
    match version {
        1 => dst.extend_from_slice(b"aaaaaaaa"),
        2 => dst.extend_from_slice(b"bbbbbbbb"),
        3 => dst.extend_from_slice(b"bbbbbbbb"),
        _ => panic!(),
    }
    dst.len()
}
