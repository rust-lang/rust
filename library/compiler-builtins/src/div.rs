use {Int, LargeInt, U64};

/// Returns `n / d`
#[no_mangle]
pub extern "C" fn __udivsi3(n: u32, d: u32) -> u32 {
    let u32_bits = u32::bits() as u32;

    // Special cases
    if d == 0 {
        panic!("Division by zero");
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

    (q << 1) | carry
}

/// Returns `n / d` and sets `*rem = n % d`
#[no_mangle]
pub extern "C" fn __udivmodsi4(a: u32, b: u32, rem: Option<&mut u32>) -> u32 {
    let d = __udivsi3(a, b);
    if let Some(rem) = rem {
        *rem = a - (d * b);
    }
    return d;
}

/// Returns `n / d` and sets `*rem = n % d`
#[no_mangle]
pub extern "C" fn __udivmoddi4(n: u64, d: u64, rem: Option<&mut u64>) -> u64 {
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

            panic!("Division by zero");
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
