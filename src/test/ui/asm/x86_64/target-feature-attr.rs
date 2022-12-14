// only-x86_64

#![feature(avx512_target_feature)]

use std::arch::asm;

#[target_feature(enable = "avx")]
unsafe fn foo() {
    let mut x = 1;
    let y = 2;
    asm!("vaddps {2:y}, {0:y}, {1:y}", in(ymm_reg) x, in(ymm_reg) y, lateout(ymm_reg) x);
    assert_eq!(x, 3);
}

unsafe fn bar() {
    let mut x = 1;
    let y = 2;
    asm!("vaddps {2:y}, {0:y}, {1:y}", in(ymm_reg) x, in(ymm_reg) y, lateout(ymm_reg) x);
    //~^ ERROR: register class `ymm_reg` requires the `avx` target feature
    //~| ERROR: register class `ymm_reg` requires the `avx` target feature
    //~| ERROR: register class `ymm_reg` requires the `avx` target feature
    assert_eq!(x, 3);
}

#[target_feature(enable = "avx512bw")]
unsafe fn baz() {
    let x = 1;
    asm!("/* {0} */", in(kreg) x);
}

unsafe fn baz2() {
    let x = 1;
    asm!("/* {0} */", in(kreg) x);
    //~^ ERROR: register class `kreg` requires at least one of the following target features: avx512bw, avx512f
}

fn main() {
    unsafe {
        foo();
        bar();
    }
}
