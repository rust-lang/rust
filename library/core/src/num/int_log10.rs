mod unchecked {
    // 0 < val <= u8::MAX
    pub const fn u8(val: u8) -> u32 {
        if val >= 100 {
            2
        } else if val >= 10 {
            1
        } else {
            0
        }
    }

    // 0 < val <= u16::MAX
    pub const fn u16(val: u16) -> u32 {
        if val >= 10_000 {
            4
        } else if val >= 1000 {
            3
        } else if val >= 100 {
            2
        } else if val >= 10 {
            1
        } else {
            0
        }
    }

    // 0 < val < 100_000_000
    const fn less_than_8(mut val: u32) -> u32 {
        let mut log = 0;
        if val >= 10_000 {
            val /= 10_000;
            log += 4;
        }
        log + if val >= 1000 {
            3
        } else if val >= 100 {
            2
        } else if val >= 10 {
            1
        } else {
            0
        }
    }

    // 0 < val <= u32::MAX
    pub const fn u32(mut val: u32) -> u32 {
        let mut log = 0;
        if val >= 100_000_000 {
            val /= 100_000_000;
            log += 8;
        }
        log + less_than_8(val)
    }

    // 0 < val < 10_000_000_000_000_000
    const fn less_than_16(mut val: u64) -> u32 {
        let mut log = 0;
        if val >= 100_000_000 {
            val /= 100_000_000;
            log += 8;
        }
        log + less_than_8(val as u32)
    }

    // 0 < val <= u64::MAX
    pub const fn u64(mut val: u64) -> u32 {
        let mut log = 0;
        if val >= 10_000_000_000_000_000 {
            val /= 10_000_000_000_000_000;
            log += 16;
        }
        log + less_than_16(val)
    }

    // 0 < val <= u128::MAX
    pub const fn u128(mut val: u128) -> u32 {
        let mut log = 0;
        if val >= 100_000_000_000_000_000_000_000_000_000_000 {
            val /= 100_000_000_000_000_000_000_000_000_000_000;
            log += 32;
            return log + less_than_8(val as u32);
        }
        if val >= 10_000_000_000_000_000 {
            val /= 10_000_000_000_000_000;
            log += 16;
        }
        log + less_than_16(val as u64)
    }

    // 0 < val <= i8::MAX
    pub const fn i8(val: i8) -> u32 {
        u8(val as u8)
    }

    // 0 < val <= i16::MAX
    pub const fn i16(val: i16) -> u32 {
        u16(val as u16)
    }

    // 0 < val <= i32::MAX
    pub const fn i32(val: i32) -> u32 {
        u32(val as u32)
    }

    // 0 < val <= i64::MAX
    pub const fn i64(val: i64) -> u32 {
        u64(val as u64)
    }

    // 0 < val <= i128::MAX
    pub const fn i128(val: i128) -> u32 {
        u128(val as u128)
    }
}

macro_rules! impl_checked {
    ($T:ident) => {
        pub const fn $T(val: $T) -> Option<u32> {
            if val > 0 { Some(unchecked::$T(val)) } else { None }
        }
    };
}

impl_checked! { u8 }
impl_checked! { u16 }
impl_checked! { u32 }
impl_checked! { u64 }
impl_checked! { u128 }
impl_checked! { i8 }
impl_checked! { i16 }
impl_checked! { i32 }
impl_checked! { i64 }
impl_checked! { i128 }
