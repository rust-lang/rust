//@ run-pass

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![allow(dead_code)]

#[rustc_layout_scalar_valid_range_start(2)]
struct U8WithTwoNiches(u8);

// 1 bytes.
enum Order1 {
    A(U8WithTwoNiches),
    B,
    C,
}

enum Order2 {
    A,
    B(U8WithTwoNiches),
    C,
}

enum Order3 {
    A,
    B,
    C(U8WithTwoNiches),
}

fn main() {
    assert_eq!(std::mem::size_of::<Order1>(), 1);
    assert_eq!(std::mem::size_of::<Order2>(), 1);
    assert_eq!(std::mem::size_of::<Order3>(), 1);
}
