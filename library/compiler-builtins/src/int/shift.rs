use int::{Int, LargeInt};

trait Ashl: Int + LargeInt {
    /// Returns `a << b`, requires `b < $ty::bits()`
    fn ashl(self, offset: u32) -> Self
        where Self: LargeInt<HighHalf = <Self as LargeInt>::LowHalf>,
    {
        let half_bits = Self::bits() / 2;
        if offset & half_bits != 0 {
            Self::from_parts(Int::zero(), self.low() << (offset - half_bits))
        } else if offset == 0 {
            self
        } else {
            Self::from_parts(self.low() << offset,
                             (self.high() << offset) |
                                (self.low() >> (half_bits - offset)))
        }
    }
}

impl Ashl for u64 {}
impl Ashl for u128 {}

trait Ashr: Int + LargeInt {
    /// Returns arithmetic `a >> b`, requires `b < $ty::bits()`
    fn ashr(self, offset: u32) -> Self
        where Self: LargeInt<LowHalf = <<Self as LargeInt>::HighHalf as Int>::UnsignedInt>,
    {
        let half_bits = Self::bits() / 2;
        if offset & half_bits != 0 {
            Self::from_parts((self.high() >> (offset - half_bits)).unsigned(),
                              self.high() >> (half_bits - 1))
        } else if offset == 0 {
            self
        } else {
            let high_unsigned = self.high().unsigned();
            Self::from_parts((high_unsigned << (half_bits - offset)) | (self.low() >> offset),
                              self.high() >> offset)
        }
    }
}

impl Ashr for i64 {}
impl Ashr for i128 {}

trait Lshr: Int + LargeInt {
    /// Returns logical `a >> b`, requires `b < $ty::bits()`
    fn lshr(self, offset: u32) -> Self
        where Self: LargeInt<HighHalf = <Self as LargeInt>::LowHalf>,
    {
        let half_bits = Self::bits() / 2;
        if offset & half_bits != 0 {
            Self::from_parts(self.high() >> (offset - half_bits), Int::zero())
        } else if offset == 0 {
            self
        } else {
            Self::from_parts((self.high() << (half_bits - offset)) |
                                (self.low() >> offset),
                             self.high() >> offset)
        }
    }
}

impl Lshr for u64 {}
impl Lshr for u128 {}

intrinsics! {
    #[cfg(not(all(feature = "c", target_arch = "x86")))]
    pub extern "C" fn __ashldi3(a: u64, b: u32) -> u64 {
        a.ashl(b)
    }

    pub extern "C" fn __ashlti3(a: u128, b: u32) -> u128 {
        a.ashl(b)
    }

    #[cfg(not(all(feature = "c", target_arch = "x86")))]
    pub extern "C" fn __ashrdi3(a: i64, b: u32) -> i64 {
        a.ashr(b)
    }

    pub extern "C" fn __ashrti3(a: i128, b: u32) -> i128 {
        a.ashr(b)
    }

    #[cfg(not(all(feature = "c", target_arch = "x86")))]
    pub extern "C" fn __lshrdi3(a: u64, b: u32) -> u64 {
        a.lshr(b)
    }

    pub extern "C" fn __lshrti3(a: u128, b: u32) -> u128 {
        a.lshr(b)
    }
}
