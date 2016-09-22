use int::{Int, LargeInt};

macro_rules! ashl {
    ($intrinsic:ident: $ty:ty) => {
        /// Returns `a << b`, requires `b < $ty::bits()`
        #[cfg_attr(not(test), no_mangle)]
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

ashl!(__ashldi3: u64);
ashr!(__ashrdi3: i64);
lshr!(__lshrdi3: u64);

#[cfg(test)]
mod tests {
    use qc::{I64, U64};

    use gcc_s;
    use quickcheck::TestResult;
    use rand;

    // NOTE We purposefully stick to `u32` for `b` here because we want "small" values (b < 64)
    quickcheck! {
        fn ashldi(a: U64, b: u32) -> TestResult {
            let a = a.0;
            if b >= 64 {
                TestResult::discard()
            } else {
                let r = super::__ashldi3(a, b);

                match gcc_s::ashldi3() {
                    Some(ashldi3) if rand::random() => {
                        TestResult::from_bool(r == unsafe { ashldi3(a, b) })
                    },
                    _ => TestResult::from_bool(r == a << b),
                }
            }
        }

        fn ashrdi(a: I64, b: u32) -> TestResult {
            let a = a.0;
            if b >= 64 {
                TestResult::discard()
            } else {
                let r = super::__ashrdi3(a, b);

                match gcc_s::ashrdi3() {
                    Some(ashrdi3) if rand::random() => {
                        TestResult::from_bool(r == unsafe { ashrdi3(a, b) })
                    },
                    _ => TestResult::from_bool(r == a >> b),
                }
            }
        }

        fn lshrdi(a: U64, b: u32) -> TestResult {
            let a = a.0;
            if b >= 64 {
                TestResult::discard()
            } else {
                let r = super::__lshrdi3(a, b);

                match gcc_s::lshrdi3() {
                    Some(lshrdi3) if rand::random() => {
                        TestResult::from_bool(r == unsafe { lshrdi3(a, b) })
                    },
                    _ => TestResult::from_bool(r == a >> b),
                }
            }
        }
    }
}
