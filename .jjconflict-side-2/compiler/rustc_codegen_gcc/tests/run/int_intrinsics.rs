// Compiler:
//
// Run-time:
#![feature(core_intrinsics, intrinsics)]
#![no_main]

use std::intrinsics::black_box;

#[rustc_intrinsic]
pub const fn ctlz<T: Copy>(_x: T) -> u32;

#[rustc_intrinsic]
pub const fn cttz<T: Copy>(_x: T) -> u32;

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    macro_rules! check {
        ($func_name:ident, $input:expr, $expected:expr, $res_ident:ident) => {{
            $res_ident += 1;
            if $func_name(black_box($input)) != $expected {
                return $res_ident;
            }
        }};
    }
    let mut res = 0;
    check!(ctlz, 0_u128, 128_u32, res);
    check!(ctlz, 1_u128, 127_u32, res);
    check!(ctlz, 0x4000_0000_0000_0000_0000_0000_0000_0000_u128, 1_u32, res);
    check!(ctlz, 0x8000_0000_0000_0000_0000_0000_0000_0000_u128, 0_u32, res);
    check!(cttz, 0_u128, 128_u32, res);
    check!(cttz, 1_u128, 0_u32, res);
    check!(cttz, 2_u128, 1_u32, res);
    check!(cttz, 0x8000_0000_0000_0000_0000_0000_0000_0000_u128, 127_u32, res);

    check!(ctlz, 0_u64, 64_u32, res);
    check!(ctlz, 1_u64, 63_u32, res);
    check!(ctlz, 0x4000_0000_0000_0000_u64, 1_u32, res);
    check!(ctlz, 0x8000_0000_0000_0000_u64, 0_u32, res);
    check!(cttz, 0_u64, 64_u32, res);
    check!(cttz, 1_u64, 0_u32, res);
    check!(cttz, 2_u64, 1_u32, res);
    check!(cttz, 0x8000_0000_0000_0000_u64, 63_u32, res);

    check!(ctlz, 0_u32, 32_u32, res);
    check!(ctlz, 1_u32, 31_u32, res);
    check!(ctlz, 0x4000_0000_u32, 1_u32, res);
    check!(ctlz, 0x8000_0000_u32, 0_u32, res);
    check!(cttz, 0_u32, 32_u32, res);
    check!(cttz, 1_u32, 0_u32, res);
    check!(cttz, 2_u32, 1_u32, res);
    check!(cttz, 0x8000_0000_u32, 31_u32, res);

    check!(ctlz, 0_u16, 16_u32, res);
    check!(ctlz, 1_u16, 15_u32, res);
    check!(ctlz, 0x4000_u16, 1_u32, res);
    check!(ctlz, 0x8000_u16, 0_u32, res);
    check!(cttz, 0_u16, 16_u32, res);
    check!(cttz, 1_u16, 0_u32, res);
    check!(cttz, 2_u16, 1_u32, res);
    check!(cttz, 0x8000_u16, 15_u32, res);

    check!(ctlz, 0_u8, 8_u32, res);
    check!(ctlz, 1_u8, 7_u32, res);
    check!(ctlz, 0x40_u8, 1_u32, res);
    check!(ctlz, 0x80_u8, 0_u32, res);
    check!(cttz, 0_u8, 8_u32, res);
    check!(cttz, 1_u8, 0_u32, res);
    check!(cttz, 2_u8, 1_u32, res);
    check!(cttz, 0x80_u8, 7_u32, res);

    0
}
