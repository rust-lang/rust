// We're testing loongarch64-specific intrinsics
//@only-target: loongarch64
#![feature(abi_unadjusted, link_llvm_intrinsics, stdarch_loongarch)]

use std::arch::loongarch64::*;

unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.crc.w.b.w"]
    fn _crc_w_b_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crc.w.h.w"]
    fn _crc_w_h_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.b.w"]
    fn _crcc_w_b_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.h.w"]
    fn _crcc_w_h_w(a: i32, b: i32) -> i32;
}

fn main() {
    test_crc_ieee();
    test_crc_castagnoli();
}

fn test_crc_ieee() {
    // crc.w.b.w: 8-bit input
    assert_eq!(crc_w_b_w(0x01, 0x00000000), 0x77073096);
    assert_eq!(unsafe { _crc_w_b_w(0x1_01, 0x00000000) }, 0x77073096); // higher bits in the first argument are ignored
    assert_eq!(crc_w_b_w(0x61, 0xffffffff_u32 as i32), 0x174841bc);
    assert_eq!(unsafe { _crc_w_b_w(0x2_61, 0xffffffff_u32 as i32) }, 0x174841bc);
    assert_eq!(crc_w_b_w(0x2a, 0x2aa1e72b), 0x772d9171);
    assert_eq!(unsafe { _crc_w_b_w(0x3_2a, 0x2aa1e72b) }, 0x772d9171);

    // crc.w.h.w: 16-bit input
    assert_eq!(crc_w_h_w(0x0001, 0x00000000), 0x191b3141);
    assert_eq!(unsafe { _crc_w_h_w(0x1_0001, 0x00000000) }, 0x191b3141);
    assert_eq!(crc_w_h_w(0x1234, 0xffffffff_u32 as i32), 0xf6b56fbf_u32 as i32);
    assert_eq!(unsafe { _crc_w_h_w(0x2_1234, 0xffffffff_u32 as i32) }, 0xf6b56fbf_u32 as i32);
    assert_eq!(crc_w_h_w(0x022b, 0x8ecec3b5_u32 as i32), 0x03a1db7c);
    assert_eq!(unsafe { _crc_w_h_w(0x3_022b, 0x8ecec3b5_u32 as i32) }, 0x03a1db7c);

    // crc.w.w.w: 32-bit input
    assert_eq!(crc_w_w_w(0x00000001, 0x00000000), 0xb8bc6765_u32 as i32);
    assert_eq!(crc_w_w_w(0x12345678, 0xffffffff_u32 as i32), 0x5092782d);
    assert_eq!(crc_w_w_w(0x00845fed, 0xae2912c8_u32 as i32), 0xc5690dd4_u32 as i32);

    // crc.w.d.w: 64-bit input
    assert_eq!(crc_w_d_w(0x0000000000000001, 0x00000000), 0xccaa009e_u32 as i32);
    assert_eq!(crc_w_d_w(0x123456789abcdef0, 0xffffffff_u32 as i32), 0xe6ddf8b5_u32 as i32);
    assert_eq!(crc_w_d_w(0xc0febeefdadafefe_u64 as i64, 0x0badeafe), 0x61a45fba);
}

fn test_crc_castagnoli() {
    // crcc.w.b.w: 8-bit input
    assert_eq!(crcc_w_b_w(0x01, 0x00000000), 0xf26b8303_u32 as i32);
    assert_eq!(unsafe { _crcc_w_b_w(0x1_01, 0x00000000) }, 0xf26b8303_u32 as i32);
    assert_eq!(crcc_w_b_w(0x61, 0xffffffff_u32 as i32), 0x3e2fbccf);
    assert_eq!(unsafe { _crcc_w_b_w(0x2_61, 0xffffffff_u32 as i32) }, 0x3e2fbccf);
    assert_eq!(crcc_w_b_w(0x2a, 0x2aa1e72b), 0xf24122e4_u32 as i32);
    assert_eq!(unsafe { _crcc_w_b_w(0x3_2a, 0x2aa1e72b) }, 0xf24122e4_u32 as i32);

    // crcc.w.h.w: 16-bit input
    assert_eq!(crcc_w_h_w(0x0001, 0x00000000), 0x13a29877);
    assert_eq!(unsafe { _crcc_w_h_w(0x1_0001, 0x00000000) }, 0x13a29877);
    assert_eq!(crcc_w_h_w(0x1234, 0xffffffff_u32 as i32), 0xf13f4cea_u32 as i32);
    assert_eq!(unsafe { _crcc_w_h_w(0x2_1234, 0xffffffff_u32 as i32) }, 0xf13f4cea_u32 as i32);
    assert_eq!(crcc_w_h_w(0x022b, 0x8ecec3b5_u32 as i32), 0x013bb2fb);
    assert_eq!(unsafe { _crcc_w_h_w(0x3_022b, 0x8ecec3b5_u32 as i32) }, 0x013bb2fb);

    // crcc.w.w.w: 32-bit input
    assert_eq!(crcc_w_w_w(0x00000001, 0x00000000), 0xdd45aab8_u32 as i32);
    assert_eq!(crcc_w_w_w(0x12345678, 0xffffffff_u32 as i32), 0x4dece20c);
    assert_eq!(crcc_w_w_w(0x00845fed, 0xae2912c8_u32 as i32), 0xffae2ed1_u32 as i32);

    // crcc.w.d.w: 64-bit input
    assert_eq!(crcc_w_d_w(0x0000000000000001, 0x00000000), 0x493c7d27);
    assert_eq!(crcc_w_d_w(0x123456789abcdef0, 0xffffffff_u32 as i32), 0xd95b664b_u32 as i32);
    assert_eq!(crcc_w_d_w(0xc0febeefdadafefe_u64 as i64, 0x0badeafe), 0x5b44f54f);
}
