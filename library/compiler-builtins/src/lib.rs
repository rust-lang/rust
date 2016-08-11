#![allow(unused_features)]
#![cfg_attr(not(test), no_std)]
#![feature(asm)]
#![feature(core_intrinsics)]
#![feature(naked_functions)]
// TODO(rust-lang/rust#35021) uncomment when that PR lands
// #![feature(rustc_builtins)]

#[cfg(test)]
extern crate core;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(test)]
mod test;

use core::mem;

/// Trait for some basic operations on integers
trait Int {
    fn bits() -> usize;
}

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl Int for u32 {
    fn bits() -> usize { 32 }
}
impl Int for i32 {
    fn bits() -> usize { 32 }
}
impl Int for u64 {
    fn bits() -> usize { 64 }
}
impl Int for i64 {
    fn bits() -> usize { 64 }
}

/// Trait to convert an integer to/from smaller parts
trait LargeInt {
    type LowHalf;
    type HighHalf;

    fn low(self) -> Self::LowHalf;
    fn high(self) -> Self::HighHalf;
    fn from_parts(low: Self::LowHalf, high: Self::HighHalf) -> Self;
}

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl LargeInt for u64 {
    type LowHalf = u32;
    type HighHalf = u32;

    fn low(self) -> u32 {
        self as u32
    }
    fn high(self) -> u32 {
        (self >> 32) as u32
    }
    fn from_parts(low: u32, high: u32) -> u64 {
        low as u64 | ((high as u64) << 32)
    }
}
impl LargeInt for i64 {
    type LowHalf = u32;
    type HighHalf = i32;

    fn low(self) -> u32 {
        self as u32
    }
    fn high(self) -> i32 {
        (self >> 32) as i32
    }
    fn from_parts(low: u32, high: i32) -> i64 {
        low as i64 | ((high as i64) << 32)
    }
}

macro_rules! absv_i2 {
    ($intrinsic:ident : $ty:ty) => {
        #[no_mangle]
        pub extern "C" fn $intrinsic(x: $ty) -> $ty {
            let n = <$ty>::bits();
            if x == 1 << (n - 1) {
                panic!();
            }
            let y = x >> (n - 1);
            (x ^ y) - y
        }

    }
}

absv_i2!(__absvsi2: i32);
absv_i2!(__absvdi2: i64);
// TODO(rust-lang/35118)?
// absv_i2!(__absvti2, i128);

#[no_mangle]
pub extern "C" fn __udivmoddi4(a: u64, b: u64, rem: *mut u64) -> u64 {
    #[cfg(target_endian = "little")]
    #[repr(C)]
    #[derive(Debug)]
    struct words {
        low: u32,
        high: u32,
    }

    #[cfg(target_endian = "big")]
    #[repr(C)]
    #[derive(Debug)]
    struct words {
        high: u32,
        low: u32,
    }

    impl words {
        fn all(&mut self) -> &mut u64 {
            unsafe { mem::transmute(self) }
        }

        fn u64(&self) -> u64 {
            unsafe { *(self as *const _ as *const u64) }
        }
    }

    impl From<u64> for words {
        fn from(x: u64) -> words {
            unsafe { mem::transmute(x) }
        }
    }

    let u32_bits = u32::bits() as u32;
    let u64_bits = u64::bits() as u32;

    let n = words::from(a);
    let d = words::from(b);

    // NOTE X is unknown, K != 0
    if n.high == 0 {
        return if d.high == 0 {
            // 0 X
            // ---
            // 0 X

            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = u64::from(n.low % d.low);
            }
            u64::from(n.low / d.low)
        } else
               // d.high != 0
               {
            // 0 X
            // ---
            // K X

            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = u64::from(n.low);
            }
            0
        };
    }

    let mut sr;
    // NOTE IMO it should be possible to leave these "uninitialized" (just declare them here)
    // because these variables get initialized below, but if I do that the compiler complains about
    // them being used before being initialized.
    let mut q = words { low: 0, high: 0 };
    let mut r = words { low: 0, high: 0 };

    // n.high != 0
    if d.low == 0 {
        if d.high == 0 {
            // K X
            // ---
            // 0 0

            // NOTE copied verbatim from compiler-rt, but does division by zero even make sense?
            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = u64::from(n.high % d.low);
            }
            return u64::from(n.high / d.low);
        }

        // d.high != 0
        if n.low == 0 {
            // K 0
            // ---
            // K 0

            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = words {
                        low: 0,
                        high: n.high % d.high,
                    }
                    .u64();
            }
            return u64::from(n.high / d.high);
        }

        // n.low != 0
        // K K
        // ---
        // K 0

        if d.high.is_power_of_two() {
            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = words {
                        low: n.low,
                        high: n.high & (d.high - 1),
                    }
                    .u64()
            }

            return u64::from(n.high >> d.high.trailing_zeros());
        }

        sr = d.high.leading_zeros().wrapping_sub(n.high.leading_zeros());

        // D > N
        if sr > u32_bits - 2 {
            if let Some(rem) = unsafe { rem.as_mut() } {
                *rem = n.u64();
            }
            return 0;
        }

        sr = sr + 1;

        // 1 <= sr <= u32_bits - 1
        // *q.all() = n.u64() << (u64_bits - sr);
        q.low = 0;
        q.high = n.low << (u32_bits - sr);
        // *r.all() = n.u64() >> sr
        r.high = n.high >> sr;
        r.low = (n.high << (u32_bits - sr)) | (n.low >> sr);
    } else
    // d.low != 0
    {
        if d.high == 0 {
            // K X
            // ---
            // 0 K
            if d.low.is_power_of_two() {
                if let Some(rem) = unsafe { rem.as_mut() } {
                    *rem = u64::from(n.low & (d.low - 1));
                }

                return if d.low == 1 {
                    n.u64()
                } else {
                    let sr = d.low.trailing_zeros();
                    words {
                            low: (n.high << (u32_bits - sr)) | (n.low >> sr),
                            high: n.high >> sr,
                        }
                        .u64()
                };
            }

            sr = 1 + u32_bits + d.low.leading_zeros() - n.high.leading_zeros();

            // 2 <= sr <= u64_bits - 1
            // *q.all() = n.u64() << (u64_bits - sr)
            // *r.all() = n.u64() >> sr;
            if sr == u32_bits {
                q.low = 0;
                q.high = n.low;
                r.high = 0;
                r.low = n.high;
            } else if sr < u32_bits
            // 2 <= sr <= u32_bits - 1
            {
                q.low = 0;
                q.high = n.low << (u32_bits - sr);
                r.high = n.high >> sr;
                r.low = (n.high << (u32_bits - sr)) | (n.low >> sr);
            } else
            // u32_bits + 1 <= sr <= u64_bits - 1
            {
                q.low = n.low << (u64_bits - sr);
                q.high = (n.high << (u64_bits - sr)) | (n.low >> (sr - u32_bits));
                r.high = 0;
                r.low = n.high >> (sr - u32_bits);
            }

        } else
        // d.high != 0
        {
            // K X
            // ---
            // K K

            sr = d.high.leading_zeros().wrapping_sub(n.high.leading_zeros());

            // D > N
            if sr > u32_bits - 1 {
                if let Some(rem) = unsafe { rem.as_mut() } {
                    *rem = a;
                    return 0;
                }
            }

            sr += 1;

            // 1 <= sr <= u32_bits
            // *q.all() = n.u64() << (u64_bits - sr)
            q.low = 0;
            if sr == u32_bits {
                q.high = n.low;
                r.high = 0;
                r.low = n.high;
            } else {
                q.high = n.low << (u32_bits - sr);
                r.high = n.high >> sr;
                r.low = (n.high << (u32_bits - sr)) | (n.low >> sr);
            }
        }
    }

    // Not a special case
    // q and r are initialized with
    // *q.all() = n.u64() << (u64_bits - sr)
    // *.r.all() = n.u64() >> sr
    // 1 <= sr <= u64_bits - 1
    let mut carry = 0;

    for _ in 0..sr {
        // r:q = ((r:q) << 1) | carry
        r.high = (r.high << 1) | (r.low >> (u32_bits - 1));
        r.low = (r.low << 1) | (q.high >> (u32_bits - 1));
        q.high = (q.high << 1) | (q.low >> (u32_bits - 1));
        q.low = (q.low << 1) | carry;

        // carry = 0
        // if r.u64() >= d.u64() {
        //     *r.all() -= d.u64();
        //     carry = 1;
        // }

        let s = (d.u64().wrapping_sub(r.u64()).wrapping_sub(1)) as i64 >> (u64_bits - 1);
        carry = (s & 1) as u32;
        *r.all() -= d.u64() & s as u64;
    }

    *q.all() = (q.u64() << 1) | carry as u64;
    if let Some(rem) = unsafe { rem.as_mut() } {
        *rem = r.u64();
    }
    q.u64()
}

#[no_mangle]
pub extern "C" fn __udivmodsi4(a: u32, b: u32, rem: *mut u32) -> u32 {
    let d = __udivsi3(a, b);
    if let Some(rem) = unsafe {rem.as_mut()} {
        *rem = a - (d*b);
    }
    return d;
}

#[no_mangle]
pub extern "C" fn __udivsi3(n: u32, d: u32) -> u32 {
    let u32_bits = u32::bits() as u32;

    // Special cases
    if d == 0 {
        return 0; // ?!
    }

    if n == 0 {
        return 0;
    }

    let mut sr = d.leading_zeros().wrapping_sub(n.leading_zeros());

    // d > n
    if sr > u32_bits - 1 {
        return 0;
    }

    // d == 1
    if sr == u32_bits - 1 {
        return n;
    }

    sr = sr + 1;

    // 1 <= sr <= u32_bits - 1
    let mut q = n << (u32_bits - sr);
    let mut r = n >> sr;

    let mut carry = 0;
    for _ in 0..sr {
        // r:q = ((r:q) << 1) | carry
        r = (r << 1) | (q >> (u32_bits - 1));
        q = (q << 1) | carry;

        // carry = 0;
        // if r > d {
        //     r -= d;
        //     carry = 1;
        // }

        let s = (d.wrapping_sub(r).wrapping_sub(1)) as i32 >> (u32_bits - 1);
        carry = (s & 1) as u32;
        r -= d & s as u32;
    }

    q = (q << 1) | carry;
    q
}
