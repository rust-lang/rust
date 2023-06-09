// run-pass
#![feature(test)]

extern crate test;
use test::black_box as b; // prevent promotion of the argument and const-propagation of the result

const BE_U32: u32 = 55u32.to_be();
const LE_U32: u32 = 55u32.to_le();

fn main() {
    assert_eq!(BE_U32, b(55u32).to_be());
    assert_eq!(LE_U32, b(55u32).to_le());

    #[cfg(not(target_os = "emscripten"))]
    {
        const BE_U128: u128 = 999999u128.to_be();
        const LE_I128: i128 = (-999999i128).to_le();
        assert_eq!(BE_U128, b(999999u128).to_be());
        assert_eq!(LE_I128, b(-999999i128).to_le());
    }
}
