//@ run-pass
//@ aux-build:cci_intrinsic.rs

extern crate cci_intrinsic;

pub fn main() {
    let val = cci_intrinsic::size_of_val(&[1u8, 2, 3]);
    assert_eq!(val, 3);
}
