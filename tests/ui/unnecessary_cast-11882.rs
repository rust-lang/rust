#![feature(const_trait_impl, const_ops)]
#![warn(clippy::unnecessary_cast)]
#![allow(unused, clippy::identity_op)]

const TEST: u64 = (!0 as u64).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const TEST2: u64 = (!0_u64 as u64).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const TEST3: u64 = (!not(!0_u64) as u64).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const TEST4: u64 = (!0 as u64 + 0).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const TEST5: u64 = (!(0 as u64 + 0)).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const TEST6: u64 = (!((0 + 0) as u64)).overflowing_shr(1_u32).0;

const CHK1: u64 = not(!0 as u64).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const CHK2: u64 = (!not(0_u64) as u64).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const CHK3: u64 = (!not(0 as u64)).overflowing_shr(1_u32).0;
//~^ unnecessary_cast
const CHK4: u64 = not(!0 as u64 + 0).overflowing_shr(1_u32).0;
//~^ unnecessary_cast

// Make sure that the calculated values aren't changed by the fixes.
const _: () = {
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST);
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST2);
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST3);
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST4);
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST5);
    assert_eq!(0x7f_ff_ff_ff_ff_ff_ff_ffu64, TEST6);
    assert_eq!(0, CHK1);
    assert_eq!(0, CHK2);
    assert_eq!(0, CHK3);
    assert_eq!(0, CHK4);
};

fn main() {
    // the non-const version of the tests
    let test: u64 = (!0 as u64).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let test2: u64 = (!0_u64 as u64).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let test3: u64 = (!not(!0_u64) as u64).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let test4: u64 = (!0 as u64 + 0).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let test5: u64 = (!(0 as u64 + 0)).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let test6: u64 = (!((0 + 0) as u64)).overflowing_shr(1_u32).0;

    let chk1: u64 = not(!0 as u64).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let chk2: u64 = (!not(0_u64) as u64).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let chk3: u64 = (!not(0 as u64)).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
    let chk4: u64 = not(!0 as u64 + 0).overflowing_shr(1_u32).0;
    //~^ unnecessary_cast
}

const fn not<T: const std::ops::Not<Output = T>>(x: T) -> T {
    !x
}
