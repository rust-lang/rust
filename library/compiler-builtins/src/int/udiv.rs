use core::intrinsics;
use int::{Int, LargeInt};

/// Returns `n / d`
#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"), not(thumbv6m))))]
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __udivsi3(n: u32, d: u32) -> u32 {
    // Special cases
    if d == 0 {
        // NOTE This should be unreachable in safe Rust because the program will panic before
        // this intrinsic is called
        unsafe {
            intrinsics::abort()
        }
    }

    if n == 0 {
        return 0;
    }

    let mut sr = d.leading_zeros().wrapping_sub(n.leading_zeros());

    // d > n
    if sr > u32::bits() - 1 {
        return 0;
    }

    // d == 1
    if sr == u32::bits() - 1 {
        return n;
    }

    sr += 1;

    // 1 <= sr <= u32::bits() - 1
    let mut q = n << (u32::bits() - sr);
    let mut r = n >> sr;

    let mut carry = 0;
    for _ in 0..sr {
        // r:q = ((r:q) << 1) | carry
        r = (r << 1) | (q >> (u32::bits() - 1));
        q = (q << 1) | carry;

        // carry = 0;
        // if r > d {
        //     r -= d;
        //     carry = 1;
        // }

        let s = (d.wrapping_sub(r).wrapping_sub(1)) as i32 >> (u32::bits() - 1);
        carry = (s & 1) as u32;
        r -= d & s as u32;
    }

    (q << 1) | carry
}

/// Returns `n % d`
#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"))))]
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __umodsi3(n: u32, d: u32) -> u32 {
    #[cfg(all(feature = "c", target_arch = "arm", not(target_os = "ios")))]
    extern "C" {
        fn __udivsi3(n: u32, d: u32) -> u32;
    }

    let q = match () {
        #[cfg(all(feature = "c", target_arch = "arm", not(target_os = "ios")))]
        () => unsafe { __udivsi3(n, d) },
        #[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"))))]
        () => __udivsi3(n, d),
    };

    n - q * d
}

/// Returns `n / d` and sets `*rem = n % d`
#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"), not(thumbv6m))))]
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __udivmodsi4(n: u32, d: u32, rem: Option<&mut u32>) -> u32 {
    #[cfg(all(feature = "c", target_arch = "arm", not(target_os = "ios")))]
    extern "C" {
        fn __udivsi3(n: u32, d: u32) -> u32;
    }

    let q = match () {
        #[cfg(all(feature = "c", target_arch = "arm", not(target_os = "ios")))]
        () => unsafe { __udivsi3(n, d) },
        #[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"))))]
        () => __udivsi3(n, d),
    };
    if let Some(rem) = rem {
        *rem = n - (q * d);
    }
    q
}

/// Returns `n / d`
#[cfg_attr(not(test), no_mangle)]
#[cfg(not(all(feature = "c", target_arch = "x86")))]
pub extern "C" fn __udivdi3(n: u64, d: u64) -> u64 {
    __udivmoddi4(n, d, None)
}

/// Returns `n % d`
#[cfg(not(all(feature = "c", target_arch = "x86")))]
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __umoddi3(a: u64, b: u64) -> u64 {
    use core::mem;

    let mut rem = unsafe { mem::uninitialized() };
    __udivmoddi4(a, b, Some(&mut rem));
    rem
}

/// Returns `n / d` and sets `*rem = n % d`
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __udivmoddi4(n: u64, d: u64, rem: Option<&mut u64>) -> u64 {
    // NOTE X is unknown, K != 0
    if n.high() == 0 {
        if d.high() == 0 {
            // 0 X
            // ---
            // 0 X

            if let Some(rem) = rem {
                *rem = u64::from(urem!(n.low(), d.low()));
            }
            return u64::from(udiv!(n.low(), d.low()));
        } else {
            // 0 X
            // ---
            // K X
            if let Some(rem) = rem {
                *rem = n;
            }
            return 0;
        };
    }

    let mut sr;
    let mut q;
    let mut r;

    if d.low() == 0 {
        if d.high() == 0 {
            // K X
            // ---
            // 0 0
            // NOTE This should be unreachable in safe Rust because the program will panic before
            // this intrinsic is called
            unsafe {
                intrinsics::abort()
            }
        }

        if n.low() == 0 {
            // K 0
            // ---
            // K 0
            if let Some(rem) = rem {
                *rem = u64::from_parts(0, urem!(n.high(), d.high()));
            }
            return u64::from(udiv!(n.high(), d.high()));
        }

        // K K
        // ---
        // K 0

        if d.high().is_power_of_two() {
            if let Some(rem) = rem {
                *rem = u64::from_parts(n.low(), n.high() & (d.high() - 1));
            }
            return u64::from(n.high() >> d.high().trailing_zeros());
        }

        sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

        // D > N
        if sr > u32::bits() - 2 {
            if let Some(rem) = rem {
                *rem = n;
            }
            return 0;
        }

        sr += 1;

        // 1 <= sr <= u32::bits() - 1
        q = n << (u64::bits() - sr);
        r = n >> sr;
    } else {
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
                    return n >> sr;
                };
            }

            sr = 1 + u32::bits() + d.low().leading_zeros() - n.high().leading_zeros();

            // 2 <= sr <= u64::bits() - 1
            q = n << (u64::bits() - sr);
            r = n >> sr;
        } else {
            // K X
            // ---
            // K K
            sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

            // D > N
            if sr > u32::bits() - 1 {
                if let Some(rem) = rem {
                    *rem = n;
                }
                return 0;
            }

            sr += 1;

            // 1 <= sr <= u32::bits()
            q = n << (u64::bits() - sr);
            r = n >> sr;
        }
    }

    // Not a special case
    // q and r are initialized with
    // q = n << (u64::bits() - sr)
    // r = n >> sr
    // 1 <= sr <= u64::bits() - 1
    let mut carry = 0;

    for _ in 0..sr {
        // r:q = ((r:q) << 1) | carry
        r = (r << 1) | (q >> (u64::bits() - 1));
        q = (q << 1) | carry as u64;

        // carry = 0
        // if r >= d {
        //     r -= d;
        //     carry = 1;
        // }
        let s = (d.wrapping_sub(r).wrapping_sub(1)) as i64 >> (u64::bits() - 1);
        carry = (s & 1) as u32;
        r -= d & s as u64;
    }

    if let Some(rem) = rem {
        *rem = r;
    }
    (q << 1) | carry as u64
}

#[cfg(test)]
mod tests {
    use qc::{U32, U64};

    check! {
        fn __udivdi3(f: extern fn(u64, u64) -> u64, n: U64, d: U64) -> Option<u64> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __umoddi3(f: extern fn(u64, u64) -> u64, n: U64, d: U64) -> Option<u64> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __udivmoddi4(f: extern fn(u64, u64, Option<&mut u64>) -> u64,
                        n: U64,
                        d: U64) -> Option<(u64, u64)> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                let mut r = 0;
                let q = f(n, d, Some(&mut r));
                Some((q, r))
            }
        }

        fn __udivsi3(f: extern fn(u32, u32) -> u32, n: U32, d: U32) -> Option<u32> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __umodsi3(f: extern fn(u32, u32) -> u32, n: U32, d: U32) -> Option<u32> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __udivmodsi4(f: extern fn(u32, u32, Option<&mut u32>) -> u32,
                        n: U32,
                        d: U32) -> Option<(u32, u32)> {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                None
            } else {
                let mut r = 0;
                let q = f(n, d, Some(&mut r));
                Some((q, r))
            }
        }
    }
}
