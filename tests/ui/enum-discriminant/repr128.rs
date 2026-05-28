//@ run-pass
#![feature(core_intrinsics, discriminant_kind)]

use std::intrinsics::discriminant_value;
use std::marker::DiscriminantKind;

#[repr(i128)]
enum Signed {
    Zero = 0,
    Staircase = 0x01_02_03_04_05_06_07_08_09_0a_0b_0c_0d_0e_0f,
    U64Limit = u64::MAX as i128 + 1,
    SmallNegative = -1,
    BigNegative = i128::MIN,
    Next,
}

#[repr(u128)]
enum Unsigned {
    Zero = 0,
    Staircase = 0x01_02_03_04_05_06_07_08_09_0a_0b_0c_0d_0e_0f,
    U64Limit = u64::MAX as u128 + 1,
    Next,
}

fn discr<T, U>(v: T, value: U)
where
    <T as DiscriminantKind>::Discriminant: PartialEq<U>,
{
    assert!(discriminant_value(&v) == value);
}

fn main() {
    discr(Signed::Zero, 0);
    discr(Signed::Staircase, 0x01_02_03_04_05_06_07_08_09_0a_0b_0c_0d_0e_0f);
    discr(Signed::U64Limit, u64::MAX as i128 + 1);
    discr(Signed::SmallNegative, -1);
    discr(Signed::BigNegative, i128::MIN);
    discr(Signed::Next, i128::MIN + 1);

    discr(Unsigned::Zero, 0);
    discr(Unsigned::Staircase, 0x01_02_03_04_05_06_07_08_09_0a_0b_0c_0d_0e_0f);
    discr(Unsigned::U64Limit, u64::MAX as u128 + 1);
    discr(Unsigned::Next, u64::MAX as u128 + 2);
}
