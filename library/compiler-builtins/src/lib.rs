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

/// Trait for some basic operations on integers
trait Int {
    fn bits() -> usize;
}

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl Int for u32 {
    fn bits() -> usize {
        32
    }
}
impl Int for i32 {
    fn bits() -> usize {
        32
    }
}
impl Int for u64 {
    fn bits() -> usize {
        64
    }
}
impl Int for i64 {
    fn bits() -> usize {
        64
    }
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

/// Return `n / d` and `*rem = n % d`
#[no_mangle]
pub extern "C" fn __udivmoddi4(n: u64, d: u64, rem: Option<&mut u64>) -> u64 {
    use core::ops::{Index, IndexMut, RangeFull};

    #[cfg(target_endian = "little")]
    #[repr(C)]
    struct U64 {
        low: u32,
        high: u32,
    }

    #[cfg(target_endian = "big")]
    #[repr(C)]
    struct U64 {
        high: u32,
        low: u32,
    }

    impl Index<RangeFull> for U64 {
        type Output = u64;

        fn index(&self, _: RangeFull) -> &u64 {
            unsafe { &*(self as *const _ as *const u64) }
        }
    }

    impl IndexMut<RangeFull> for U64 {
        fn index_mut(&mut self, _: RangeFull) -> &mut u64 {
            unsafe { &mut *(self as *const _ as *mut u64) }
        }
    }

    let u32_bits = u32::bits() as u32;
    let u64_bits = u64::bits() as u32;

    // NOTE X is unknown, K != 0
    if n.high() == 0 {
        if d.high() == 0 {
            // 0 X
            // ---
            // 0 X

            if let Some(rem) = rem {
                *rem = u64::from(n.low() % d.low());
            }
            return u64::from(n.low() / d.low());
        } else
        // d.high() != 0
        {
            // 0 X
            // ---
            // K X

            if let Some(rem) = rem {
                *rem = u64::from(n.low());
            }
            return 0;
        };
    }

    let mut sr;
    let mut q = U64 { low: 0, high: 0 };
    let mut r = U64 { low: 0, high: 0 };

    // n.high() != 0
    if d.low() == 0 {
        if d.high() == 0 {
            // K X
            // ---
            // 0 0

            // NOTE copied verbatim from compiler-rt. This probably lets the intrinsic decide how to
            // handle the division by zero (SIGFPE, 0, etc.). But this part shouldn't be reachable
            // from safe code.
            if let Some(rem) = rem {
                *rem = u64::from(n.high() % d.low());
            }
            return u64::from(n.high() / d.low());
        }

        // d.high() != 0
        if n.low() == 0 {
            // K 0
            // ---
            // K 0

            if let Some(rem) = rem {
                *rem = U64 {
                    low: 0,
                    high: n.high() % d.high(),
                }[..];
            }
            return u64::from(n.high() / d.high());
        }

        // n.low() != 0
        // K K
        // ---
        // K 0

        if d.high().is_power_of_two() {
            if let Some(rem) = rem {
                *rem = U64 {
                    low: n.low(),
                    high: n.high() & (d.high() - 1),
                }[..];
            }

            return u64::from(n.high() >> d.high().trailing_zeros());
        }

        sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

        // D > N
        if sr > u32_bits - 2 {
            if let Some(rem) = rem {
                *rem = n;
            }
            return 0;
        }

        sr = sr + 1;

        // 1 <= sr <= u32_bits - 1
        // q = n << (u64_bits - sr);
        q.low = 0;
        q.high = n.low() << (u32_bits - sr);
        // r = n >> sr
        r.high = n.high() >> sr;
        r.low = (n.high() << (u32_bits - sr)) | (n.low() >> sr);
    } else
    // d.low() != 0
    {
        if d.high() == 0 {
            // K X
            // ---
            // 0 K
            if d.low().is_power_of_two() {
                if let Some(rem) = rem {
                    *rem = u64::from(n.low() & (d.low() - 1));
                }

                if d.low() == 1 {
                    return n;
                } else {
                    let sr = d.low().trailing_zeros();
                    return U64 {
                        low: (n.high() << (u32_bits - sr)) | (n.low() >> sr),
                        high: n.high() >> sr,
                    }[..];
                };
            }

            sr = 1 + u32_bits + d.low().leading_zeros() - n.high().leading_zeros();

            // 2 <= sr <= u64_bits - 1
            // q = n << (u64_bits - sr)
            // r = n >> sr;
            if sr == u32_bits {
                q.low = 0;
                q.high = n.low();
                r.high = 0;
                r.low = n.high();
            } else if sr < u32_bits
            // 2 <= sr <= u32_bits - 1
            {
                q.low = 0;
                q.high = n.low() << (u32_bits - sr);
                r.high = n.high() >> sr;
                r.low = (n.high() << (u32_bits - sr)) | (n.low() >> sr);
            } else
            // u32_bits + 1 <= sr <= u64_bits - 1
            {
                q.low = n.low() << (u64_bits - sr);
                q.high = (n.high() << (u64_bits - sr)) | (n.low() >> (sr - u32_bits));
                r.high = 0;
                r.low = n.high() >> (sr - u32_bits);
            }

        } else
        // d.high() != 0
        {
            // K X
            // ---
            // K K

            sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

            // D > N
            if sr > u32_bits - 1 {
                if let Some(rem) = rem {
                    *rem = n;
                    return 0;
                }
            }

            sr += 1;

            // 1 <= sr <= u32_bits
            // q = n << (u64_bits - sr)
            q.low = 0;
            if sr == u32_bits {
                q.high = n.low();
                r.high = 0;
                r.low = n.high();
            } else {
                q.high = n.low() << (u32_bits - sr);
                r.high = n.high() >> sr;
                r.low = (n.high() << (u32_bits - sr)) | (n.low() >> sr);
            }
        }
    }

    // Not a special case
    // q and r are initialized with
    // q = n << (u64_bits - sr)
    // r = n >> sr
    // 1 <= sr <= u64_bits - 1
    let mut carry = 0;

    for _ in 0..sr {
        // r:q = ((r:q) << 1) | carry
        r[..] = (r[..] << 1) | (q[..] >> 63);
        q[..] = (q[..] << 1) | carry as u64;

        // carry = 0
        // if r >= d {
        //     r -= d;
        //     carry = 1;
        // }

        let s = (d.wrapping_sub(r[..]).wrapping_sub(1)) as i64 >> (u64_bits - 1);
        carry = (s & 1) as u32;
        r[..] -= d & s as u64;
    }

    q[..] = (q[..] << 1) | carry as u64;
    if let Some(rem) = rem {
        *rem = r[..];
    }
    q[..]
}

/// Return `n / d` and `*rem = n % d`
#[no_mangle]
pub extern "C" fn __udivmodsi4(a: u32, b: u32, rem: Option<&mut u32>) -> u32 {
    let d = __udivsi3(a, b);
    if let Some(rem) = rem {
        *rem = a - (d * b);
    }
    return d;
}

/// Return `n / d`
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
