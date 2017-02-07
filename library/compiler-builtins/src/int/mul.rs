use int::LargeInt;
use int::Int;

macro_rules! mul {
    ($(#[$attr:meta])+ |
     $abi:tt, $intrinsic:ident: $ty:ty) => {
        /// Returns `a * b`
        $(#[$attr])+
        pub extern $abi fn $intrinsic(a: $ty, b: $ty) -> $ty {
            let half_bits = <$ty>::bits() / 4;
            let lower_mask = !0 >> half_bits;
            let mut low = (a.low() & lower_mask).wrapping_mul(b.low() & lower_mask);
            let mut t = low >> half_bits;
            low &= lower_mask;
            t += (a.low() >> half_bits).wrapping_mul(b.low() & lower_mask);
            low += (t & lower_mask) << half_bits;
            let mut high = (t >> half_bits) as hty!($ty);
            t = low >> half_bits;
            low &= lower_mask;
            t += (b.low() >> half_bits).wrapping_mul(a.low() & lower_mask);
            low += (t & lower_mask) << half_bits;
            high += (t >> half_bits) as hty!($ty);
            high += (a.low() >> half_bits).wrapping_mul(b.low() >> half_bits) as hty!($ty);
            high = high.wrapping_add(a.high().wrapping_mul(b.low() as hty!($ty)))
                       .wrapping_add((a.low() as hty!($ty)).wrapping_mul(b.high()));
            <$ty>::from_parts(low, high)
        }
    }
}

macro_rules! mulo {
    ($intrinsic:ident: $ty:ty) => {
        // Default is "C" ABI
        mulo!($intrinsic: $ty, "C");
    };
    ($intrinsic:ident: $ty:ty, $abi:tt) => {
        /// Returns `a * b` and sets `*overflow = 1` if `a * b` overflows
        #[cfg_attr(not(test), no_mangle)]
        pub extern $abi fn $intrinsic(a: $ty, b: $ty, overflow: &mut i32) -> $ty {
            *overflow = 0;
            let result = a.wrapping_mul(b);
            if a == <$ty>::min_value() {
                if b != 0 && b != 1 {
                    *overflow = 1;
                }
                return result;
            }
            if b == <$ty>::min_value() {
                if a != 0 && a != 1 {
                    *overflow = 1;
                }
                return result;
            }

            let sa = a >> (<$ty>::bits() - 1);
            let abs_a = (a ^ sa) - sa;
            let sb = b >> (<$ty>::bits() - 1);
            let abs_b = (b ^ sb) - sb;
            if abs_a < 2 || abs_b < 2 {
                return result;
            }
            if sa == sb {
                if abs_a > <$ty>::max_value() / abs_b {
                    *overflow = 1;
                }
            } else {
                if abs_a > <$ty>::min_value() / -abs_b {
                    *overflow = 1;
                }
            }
            result
        }
    }
}

#[cfg(not(all(feature = "c", target_arch = "x86")))]
mul!(#[cfg_attr(all(not(test), not(target_arch = "arm")), no_mangle)]
     #[cfg_attr(all(not(test), target_arch = "arm"), inline(always))]
     | "C", __muldi3: u64);

#[cfg(not(target_arch = "arm"))]
mul!(#[cfg_attr(not(test), no_mangle)]
     | "C", __multi3: i128);

#[cfg(target_arch = "arm")]
mul!(#[cfg_attr(not(test), no_mangle)]
     | "aapcs", __multi3: i128);

mulo!(__mulosi4: i32);
mulo!(__mulodi4: i64);

#[cfg(all(windows, target_pointer_width="64"))]
mulo!(__muloti4: i128, "unadjusted");
#[cfg(not(all(windows, target_pointer_width="64")))]
mulo!(__muloti4: i128);

#[cfg(test)]
mod tests {
    use qc::{I32, I64, U64};

    check! {
        fn __muldi3(f: extern fn(u64, u64) -> u64, a: U64, b: U64)
                    -> Option<u64> {
            Some(f(a.0, b.0))
        }

        fn __mulosi4(f: extern fn(i32, i32, &mut i32) -> i32,
                     a: I32,
                     b: I32) -> Option<(i32, i32)> {
            let (a, b) = (a.0, b.0);
            let mut overflow = 2;
            let r = f(a, b, &mut overflow);
            if overflow != 0 && overflow != 1 {
                panic!("Invalid value {} for overflow", overflow);
            }
            Some((r, overflow))
        }

        fn __mulodi4(f: extern fn(i64, i64, &mut i32) -> i64,
                     a: I64,
                     b: I64) -> Option<(i64, i32)> {
            let (a, b) = (a.0, b.0);
            let mut overflow = 2;
            let r = f(a, b, &mut overflow);
            if overflow != 0 && overflow != 1 {
                panic!("Invalid value {} for overflow", overflow);
            }
            Some((r, overflow))
        }
    }
}

#[cfg(test)]
#[cfg(all(not(windows),
          not(target_arch = "mips64"),
          not(target_arch = "mips64el"),
          target_pointer_width="64"))]
mod tests_i128 {
    use qc::I128;

    check! {
        fn __multi3(f: extern fn(i128, i128) -> i128, a: I128, b: I128)
                    -> Option<i128> {
            Some(f(a.0, b.0))
        }
        fn __muloti4(f: extern fn(i128, i128, &mut i32) -> i128,
                     a: I128,
                     b: I128) -> Option<(i128, i32)> {
            let (a, b) = (a.0, b.0);
            let mut overflow = 2;
            let r = f(a, b, &mut overflow);
            if overflow != 0 && overflow != 1 {
                panic!("Invalid value {} for overflow", overflow);
            }
            Some((r, overflow))
        }
    }
}
