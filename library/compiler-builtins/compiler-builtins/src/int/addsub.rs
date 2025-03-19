use crate::int::{DInt, Int, MinInt};

trait UAddSub: DInt + Int {
    fn uadd(self, other: Self) -> Self {
        let (lo, carry) = self.lo().overflowing_add(other.lo());
        let hi = self.hi().wrapping_add(other.hi());
        let carry = if carry { Self::H::ONE } else { Self::H::ZERO };
        Self::from_lo_hi(lo, hi.wrapping_add(carry))
    }
    fn uadd_one(self) -> Self {
        let (lo, carry) = self.lo().overflowing_add(Self::H::ONE);
        let carry = if carry { Self::H::ONE } else { Self::H::ZERO };
        Self::from_lo_hi(lo, self.hi().wrapping_add(carry))
    }
    fn usub(self, other: Self) -> Self {
        let uneg = (!other).uadd_one();
        self.uadd(uneg)
    }
}

impl UAddSub for u128 {}

trait AddSub: Int
where
    <Self as MinInt>::UnsignedInt: UAddSub,
{
    fn add(self, other: Self) -> Self {
        Self::from_unsigned(self.unsigned().uadd(other.unsigned()))
    }
    fn sub(self, other: Self) -> Self {
        Self::from_unsigned(self.unsigned().usub(other.unsigned()))
    }
}

impl AddSub for u128 {}
impl AddSub for i128 {}

trait Addo: AddSub
where
    <Self as MinInt>::UnsignedInt: UAddSub,
{
    fn addo(self, other: Self) -> (Self, bool) {
        let sum = AddSub::add(self, other);
        (sum, (other < Self::ZERO) != (sum < self))
    }
}

impl Addo for i128 {}
impl Addo for u128 {}

trait Subo: AddSub
where
    <Self as MinInt>::UnsignedInt: UAddSub,
{
    fn subo(self, other: Self) -> (Self, bool) {
        let sum = AddSub::sub(self, other);
        (sum, (other < Self::ZERO) != (self < sum))
    }
}

impl Subo for i128 {}
impl Subo for u128 {}

intrinsics! {
    pub extern "C" fn __rust_i128_add(a: i128, b: i128) -> i128 {
        AddSub::add(a,b)
    }

    pub extern "C" fn __rust_i128_addo(a: i128, b: i128, oflow: &mut i32) -> i128 {
        let (add, o) = a.addo(b);
        *oflow = o.into();
        add
    }

    pub extern "C" fn __rust_u128_add(a: u128, b: u128) -> u128 {
        AddSub::add(a,b)
    }

    pub extern "C" fn __rust_u128_addo(a: u128, b: u128, oflow: &mut i32) -> u128 {
        let (add, o) = a.addo(b);
        *oflow = o.into();
        add
    }

    pub extern "C" fn __rust_i128_sub(a: i128, b: i128) -> i128 {
        AddSub::sub(a,b)
    }

    pub extern "C" fn __rust_i128_subo(a: i128, b: i128, oflow: &mut i32) -> i128 {
        let (sub, o) = a.subo(b);
        *oflow = o.into();
        sub
    }

    pub extern "C" fn __rust_u128_sub(a: u128, b: u128) -> u128 {
        AddSub::sub(a,b)
    }

    pub extern "C" fn __rust_u128_subo(a: u128, b: u128, oflow: &mut i32) -> u128 {
        let (sub, o) = a.subo(b);
        *oflow = o.into();
        sub
    }
}
