use int::{DInt, HInt, Int};

trait Ashl: DInt {
    /// Returns `a << b`, requires `b < Self::BITS`
    fn ashl(self, shl: u32) -> Self {
        let n_h = Self::H::BITS;
        if shl & n_h != 0 {
            // we only need `self.lo()` because `self.hi()` will be shifted out entirely
            self.lo().wrapping_shl(shl - n_h).widen_hi()
        } else if shl == 0 {
            self
        } else {
            Self::from_lo_hi(
                self.lo().wrapping_shl(shl),
                self.lo().logical_shr(n_h - shl) | self.hi().wrapping_shl(shl),
            )
        }
    }
}

impl Ashl for u32 {}
impl Ashl for u64 {}
impl Ashl for u128 {}

trait Ashr: DInt {
    /// Returns arithmetic `a >> b`, requires `b < Self::BITS`
    fn ashr(self, shr: u32) -> Self {
        let n_h = Self::H::BITS;
        if shr & n_h != 0 {
            Self::from_lo_hi(
                self.hi().wrapping_shr(shr - n_h),
                // smear the sign bit
                self.hi().wrapping_shr(n_h - 1),
            )
        } else if shr == 0 {
            self
        } else {
            Self::from_lo_hi(
                self.lo().logical_shr(shr) | self.hi().wrapping_shl(n_h - shr),
                self.hi().wrapping_shr(shr),
            )
        }
    }
}

impl Ashr for i32 {}
impl Ashr for i64 {}
impl Ashr for i128 {}

trait Lshr: DInt {
    /// Returns logical `a >> b`, requires `b < Self::BITS`
    fn lshr(self, shr: u32) -> Self {
        let n_h = Self::H::BITS;
        if shr & n_h != 0 {
            self.hi().logical_shr(shr - n_h).zero_widen()
        } else if shr == 0 {
            self
        } else {
            Self::from_lo_hi(
                self.lo().logical_shr(shr) | self.hi().wrapping_shl(n_h - shr),
                self.hi().logical_shr(shr),
            )
        }
    }
}

impl Lshr for u32 {}
impl Lshr for u64 {}
impl Lshr for u128 {}

intrinsics! {
    #[maybe_use_optimized_c_shim]
    pub extern "C" fn __ashlsi3(a: u32, b: u32) -> u32 {
        a.ashl(b)
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_llsl]
    pub extern "C" fn __ashldi3(a: u64, b: u32) -> u64 {
        a.ashl(b)
    }

    pub extern "C" fn __ashlti3(a: u128, b: u32) -> u128 {
        a.ashl(b)
    }

    #[maybe_use_optimized_c_shim]
    pub extern "C" fn __ashrsi3(a: i32, b: u32) -> i32 {
        a.ashr(b)
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_lasr]
    pub extern "C" fn __ashrdi3(a: i64, b: u32) -> i64 {
        a.ashr(b)
    }

    pub extern "C" fn __ashrti3(a: i128, b: u32) -> i128 {
        a.ashr(b)
    }

    #[maybe_use_optimized_c_shim]
    pub extern "C" fn __lshrsi3(a: u32, b: u32) -> u32 {
        a.lshr(b)
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_llsr]
    pub extern "C" fn __lshrdi3(a: u64, b: u32) -> u64 {
        a.lshr(b)
    }

    pub extern "C" fn __lshrti3(a: u128, b: u32) -> u128 {
        a.lshr(b)
    }
}
