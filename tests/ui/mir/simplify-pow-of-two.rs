// run-pass

#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

fn slow_2_u(a: u32) -> u32 {
    2u32.pow(a)
}

fn slow_2_i(a: u32) -> i32 {
    2i32.pow(a)
}

fn slow_4_u(a: u32) -> u32 {
    4u32.pow(a)
}

fn slow_4_i(a: u32) -> i32 {
    4i32.pow(a)
}

fn slow_256_u(a: u32) -> u32 {
    256u32.pow(a)
}

fn slow_256_i(a: u32) -> i32 {
    256i32.pow(a)
}

fn main() {
    assert_eq!(slow_2_u(0), 1);
    assert_eq!(slow_2_i(0), 1);
    assert_eq!(slow_2_u(1), 2);
    assert_eq!(slow_2_i(1), 2);
    assert_eq!(slow_2_u(2), 4);
    assert_eq!(slow_2_i(2), 4);
    assert_eq!(slow_4_u(4), 256);
    assert_eq!(slow_4_i(4), 256);
    assert_eq!(slow_4_u(15), 1073741824);
    assert_eq!(slow_4_i(15), 1073741824);
    assert_eq!(slow_4_u(16), 0);
    assert_eq!(slow_4_i(16), 0);
    assert_eq!(slow_4_u(17), 0);
    assert_eq!(slow_4_i(17), 0);
    assert_eq!(slow_256_u(2), 65536);
    assert_eq!(slow_256_i(2), 65536);

    for i in 0..300 {
        for j in 0..3000 {
            let ix = 2u128.pow(i);
            assert_eq!(ix.pow(j), test_mir(i, j), "{ix}, {j}");
        }
    }
}

/// num is the power used to get recv, will be calculated while building this
/// MIR but it's necessary here for testing
///
/// You can test this out in the playground here:
/// https://play.rust-lang.org/?version=nightly&mode=release&edition=2021&gist=de34e2a6a8f9114ce01bfb62f9379413
///
/// This equals `ix.pow(j)` both with and without this optimization.
#[custom_mir(dialect = "built")]
pub fn test_mir(num: u32, exp: u32) -> u128 {
    mir! {
        {
            let num_shl = Checked(exp * num);
            let shl_result = num_shl.0 < 128;
            let shl = 1u128 << num_shl.0;
            let fine_bool = shl_result | num_shl.1;
            let fine = fine_bool as u128;
            RET = shl * fine;
            Return()
        }
    }
}
