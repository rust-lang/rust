//@ run-pass
#![allow(improper_ctypes)]

// MSVC doesn't support 128 bit integers, and other Windows
// C compilers have very inconsistent views on how the ABI
// should look like.

//@ ignore-windows
//@ ignore-32bit

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn identity(f: u128) -> u128;
    fn square(f: i128) -> i128;
    fn sub(f: i128, f: i128) -> i128;
}

fn main() {
    unsafe {
        let a = 0x734C_C2F2_A521;
        let b = 0x33EE_0E2A_54E2_59DA_A0E7_8E41;
        let b_out = identity(b);
        assert_eq!(b, b_out);
        let a_square = square(a);
        assert_eq!(b, a_square as u128);
        let k = 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210;
        let k_d = 0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420;
        let k_out = sub(k_d, k);
        assert_eq!(k, k_out);
    }
}
