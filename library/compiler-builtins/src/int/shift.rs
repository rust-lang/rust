use int::{Int, LargeInt};

macro_rules! ashl {
    ($intrinsic:ident: $ty:ty) => {
        /// Returns `a << b`, requires `b < $ty::bits()`
        #[cfg_attr(not(test), no_mangle)]
        #[cfg_attr(all(not(test), not(target_arch = "arm")), no_mangle)]
        #[cfg_attr(all(not(test), target_arch = "arm"), inline(always))]
        pub extern "C" fn $intrinsic(a: $ty, b: u32) -> $ty {
            let half_bits = <$ty>::bits() / 2;
            if b & half_bits != 0 {
                <$ty>::from_parts(0, a.low() << (b - half_bits))
            } else if b == 0 {
                a
            } else {
                <$ty>::from_parts(a.low() << b, (a.high() << b) | (a.low() >> (half_bits - b)))
            }
        }
    }
}

macro_rules! ashr {
    ($intrinsic:ident: $ty:ty) => {
        /// Returns arithmetic `a >> b`, requires `b < $ty::bits()`
        #[cfg_attr(not(test), no_mangle)]
        #[cfg_attr(all(not(test), not(target_arch = "arm")), no_mangle)]
        #[cfg_attr(all(not(test), target_arch = "arm"), inline(always))]
        pub extern "C" fn $intrinsic(a: $ty, b: u32) -> $ty {
            let half_bits = <$ty>::bits() / 2;
            if b & half_bits != 0 {
                <$ty>::from_parts((a.high() >> (b - half_bits)) as <$ty as LargeInt>::LowHalf,
                                  a.high() >> (half_bits - 1))
            } else if b == 0 {
                a
            } else {
                let high_unsigned = a.high() as <$ty as LargeInt>::LowHalf;
                <$ty>::from_parts((high_unsigned << (half_bits - b)) | (a.low() >> b),
                                  a.high() >> b)
            }
        }
    }
}

macro_rules! lshr {
    ($intrinsic:ident: $ty:ty) => {
        /// Returns logical `a >> b`, requires `b < $ty::bits()`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: u32) -> $ty {
            let half_bits = <$ty>::bits() / 2;
            if b & half_bits != 0 {
                <$ty>::from_parts(a.high() >> (b - half_bits), 0)
            } else if b == 0 {
                a
            } else {
                <$ty>::from_parts((a.high() << (half_bits - b)) | (a.low() >> b), a.high() >> b)
            }
        }
    }
}

#[cfg(not(all(feature = "c", target_arch = "x86")))]
ashl!(__ashldi3: u64);

ashl!(__ashlti3: u128);

#[cfg(not(all(feature = "c", target_arch = "x86")))]
ashr!(__ashrdi3: i64);

ashr!(__ashrti3: i128);

#[cfg(not(all(feature = "c", target_arch = "x86")))]
lshr!(__lshrdi3: u64);

lshr!(__lshrti3: u128);
