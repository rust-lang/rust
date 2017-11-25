use int::LargeInt;

trait Sub: LargeInt {
    fn sub(self, other: Self) -> Self {
        let neg_other = (!other).wrapping_add(Self::ONE);
        self.wrapping_add(neg_other)
    }
}

impl Sub for i128 {}
impl Sub for u128 {}

trait Subo: Sub {
    fn subo(self, other: Self, overflow: &mut i32) -> Self {
        *overflow = 0;
        let result = Sub::sub(self, other);
        if other >= Self::ZERO {
            if result > self {
                *overflow = 1;
            }
        } else {
            if result <= self {
                *overflow = 1;
            }
        }
        result
    }
}

impl Subo for i128 {}
impl Subo for u128 {}

#[cfg_attr(not(stage0), lang = "i128_sub")]
pub fn rust_i128_sub(a: i128, b: i128) -> i128 {
    rust_u128_sub(a as _, b as _) as _
}
#[cfg_attr(not(stage0), lang = "i128_subo")]
pub fn rust_i128_subo(a: i128, b: i128) -> (i128, bool) {
    let mut oflow = 0;
    let r = a.subo(b, &mut oflow);
    (r, oflow != 0)
}
#[cfg_attr(not(stage0), lang = "u128_sub")]
pub fn rust_u128_sub(a: u128, b: u128) -> u128 {
    a.sub(b)
}
#[cfg_attr(not(stage0), lang = "u128_subo")]
pub fn rust_u128_subo(a: u128, b: u128) -> (u128, bool) {
    let mut oflow = 0;
    let r = a.subo(b, &mut oflow);
    (r, oflow != 0)
}
