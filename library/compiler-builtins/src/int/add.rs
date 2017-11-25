use int::LargeInt;
use int::Int;

trait UAdd: LargeInt {
    fn uadd(self, other: Self) -> Self {
        let (low, carry) = self.low().overflowing_add(other.low());
        let high = self.high().wrapping_add(other.high());
        let carry = if carry { Self::HighHalf::ONE } else { Self::HighHalf::ZERO };
        Self::from_parts(low, high.wrapping_add(carry))
    }
}

impl UAdd for u128 {}

trait Add: Int
    where <Self as Int>::UnsignedInt: UAdd
{
    fn add(self, other: Self) -> Self {
        Self::from_unsigned(self.unsigned().uadd(other.unsigned()))
    }
}

impl Add for u128 {}
impl Add for i128 {}

trait Addo: Add
    where <Self as Int>::UnsignedInt: UAdd
{
    fn addo(self, other: Self, overflow: &mut i32) -> Self {
        *overflow = 0;
        let result = Add::add(self, other);
        if other >= Self::ZERO {
            if result < self {
                *overflow = 1;
            }
        } else {
            if result >= self {
                *overflow = 1;
            }
        }
        result
    }
}

impl Addo for i128 {}
impl Addo for u128 {}

#[cfg_attr(not(stage0), lang = "i128_add")]
pub fn rust_i128_add(a: i128, b: i128) -> i128 {
    rust_u128_add(a as _, b as _) as _
}
#[cfg_attr(not(stage0), lang = "i128_addo")]
pub fn rust_i128_addo(a: i128, b: i128) -> (i128, bool) {
    let mut oflow = 0;
    let r = a.addo(b, &mut oflow);
    (r, oflow != 0)
}
#[cfg_attr(not(stage0), lang = "u128_add")]
pub fn rust_u128_add(a: u128, b: u128) -> u128 {
    a.add(b)
}
#[cfg_attr(not(stage0), lang = "u128_addo")]
pub fn rust_u128_addo(a: u128, b: u128) -> (u128, bool) {
    let mut oflow = 0;
    let r = a.addo(b, &mut oflow);
    (r, oflow != 0)
}
