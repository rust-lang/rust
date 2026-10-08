/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/log2f.c and src/math/log2f_data.c (Arm
 * optimized-routines, which glibc uses too)
 * Copyright (c) 2017-2018, Arm Limited.
 */

const TABLE_BITS: u32 = 4;
const N: usize = 1 << TABLE_BITS;
const OFF: u32 = 0x3f330000;

/* ULP error: 0.752 (nearest rounding.)
 * Relative error: 1.9 * 2^-26 (before rounding.)
 * T[i] = (1/c, log2(c)) for the subinterval i of [OFF, 2*OFF].
 */
#[rustfmt::skip]
const T: [(f64, f64); N] = [
    (hf64!("0x1.661ec79f8f3bep+0"), hf64!("-0x1.efec65b963019p-2")),
    (hf64!("0x1.571ed4aaf883dp+0"), hf64!("-0x1.b0b6832d4fca4p-2")),
    (hf64!("0x1.49539f0f010bp+0"), hf64!("-0x1.7418b0a1fb77bp-2")),
    (hf64!("0x1.3c995b0b80385p+0"), hf64!("-0x1.39de91a6dcf7bp-2")),
    (hf64!("0x1.30d190c8864a5p+0"), hf64!("-0x1.01d9bf3f2b631p-2")),
    (hf64!("0x1.25e227b0b8eap+0"), hf64!("-0x1.97c1d1b3b7afp-3")),
    (hf64!("0x1.1bb4a4a1a343fp+0"), hf64!("-0x1.2f9e393af3c9fp-3")),
    (hf64!("0x1.12358f08ae5bap+0"), hf64!("-0x1.960cbbf788d5cp-4")),
    (hf64!("0x1.0953f419900a7p+0"), hf64!("-0x1.a6f9db6475fcep-5")),
    (hf64!("0x1p+0"), hf64!("0x0p+0")),
    (hf64!("0x1.e608cfd9a47acp-1"), hf64!("0x1.338ca9f24f53dp-4")),
    (hf64!("0x1.ca4b31f026aap-1"), hf64!("0x1.476a9543891bap-3")),
    (hf64!("0x1.b2036576afce6p-1"), hf64!("0x1.e840b4ac4e4d2p-3")),
    (hf64!("0x1.9c2d163a1aa2dp-1"), hf64!("0x1.40645f0c6651cp-2")),
    (hf64!("0x1.886e6037841edp-1"), hf64!("0x1.88e9c2c1b9ff8p-2")),
    (hf64!("0x1.767dcf5534862p-1"), hf64!("0x1.ce0a44eb17bccp-2")),
];

const A: [f64; 4] = [
    hf64!("-0x1.712b6f70a7e4dp-2"),
    hf64!("0x1.ecabf496832ep-2"),
    hf64!("-0x1.715479ffae3dep-1"),
    hf64!("0x1.715475f35c8b8p0"),
];

/// The base 2 logarithm of `x` (f32).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn log2f(x: f32) -> f32 {
    let mut ix = x.to_bits();
    /* Fix sign of zero with downward rounding when x==1. */
    if ix == 0x3f800000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x00800000) >= 0x7f800000 - 0x00800000 {
        /* x < 0x1p-126 or inf or nan. */
        if ix << 1 == 0 {
            return -1.0 / (x * x); /* log2(+-0) = -inf */
        }
        if ix == 0x7f800000 {
            return x; /* log2(inf) = inf */
        }
        if (ix >> 31) != 0 || ix << 1 >= 0xff000000 {
            return (x - x) / (x - x); /* log2(-#) = NaN */
        }
        /* x is subnormal, normalize it. */
        ix = (x * hf32!("0x1p23")).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }

    /* x = 2^k z; where z is in range [OFF,2*OFF] and exact.
     * The range is split into N subintervals.
     * The ith subinterval contains z and c is near its center. */
    let tmp = ix.wrapping_sub(OFF);
    let i = (tmp >> (23 - TABLE_BITS)) as usize % N;
    let top = tmp & 0xff800000;
    let iz = ix.wrapping_sub(top);
    let k = (tmp as i32) >> 23; /* arithmetic shift */
    let (invc, logc) = i!(T, i);
    let z = f32::from_bits(iz) as f64;

    /* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k */
    let r = z * invc - 1.0;
    let y0 = logc + k as f64;

    /* Pipelined polynomial evaluation to approximate log1p(r)/ln2. */
    let r2 = r * r;
    let mut y = A[1] * r + A[2];
    y = A[0] * r2 + y;
    let p = A[3] * r + y0;
    y = y * r2 + p;
    y as f32
}
