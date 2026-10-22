/* SPDX-License-Identifier: MIT */
/* origin: core-math/src/binary32/exp2m1/exp2m1f.c
 * Copyright (c) 2022-2025 Alexei Sibidanov.
 * Ported to Rust in 2026, malezjaa
 * Approximate CORE-MATH commit: 8ea8ea35c518d06b72de4cf1d528e7a2d16f104c
 */

use super::support::{CastFrom, CastInto, Float, Int, cold_path};

/// Base-2 exponential minus one (f32).
///
/// Calculate `2^x - 1`, that is, 2 raised to the power `x` minus `1`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn exp2m1f(x: f32) -> f32 {
    cr_exp2m1f(x)
}

fn cr_exp2m1f(x: f32) -> f32 {
    const Q: [(f32, f32); 3] = [
        (hf32!("0x1.fffffep127"), hf32!("0x1.fffffep127")),
        (hf32!("0x1.fffffep127"), hf32!("0x1p+103")),
        (-1.0, hf32!("0x1p-26")),
    ];

    let z = f64::from(x);
    let ux = x.to_bits();
    let ax = ux & u32::MAX >> 1;

    // x <= -25
    if ux >= 0xc1c8_0000 {
        cold_path();
        if ax > (0xff << 23) {
            return x + x; // NaN
        }
        // avoid spurious inexact exception for -Inf
        if ux == 0xff800000 {
            return Q[2].0;
        } else {
            return Q[2].0 + Q[2].1;
        }
    }

    // x >= 128, result is infinite (with default rounding)
    if ax >= 0x4300_0000 {
        cold_path();
        if ax > (0xff << 23) {
            return x + x; // nan
        }
        // for x=128 and rounding downward or to zero, there is no overflow
        let special = usize::from(x == 128.0 && (Q[1].0 + Q[1].1 == Q[1].0));
        // avoid spurious inexact exception for +Inf
        if ux == 0x7f800000 {
            return x;
        } else {
            return Q[special].0 + Q[special].1;
        }
    }

    if ax < 0x3df95f1f {
        // |x| < 8.44e-2/log(2)
        cold_path();
        let z2 = z * z;
        let mut r: f64;

        if ax < 0x3d67a4cc {
            // |x| < 3.92e-2/log(2)
            cold_path();
            if ax < 0x3caa2fee {
                // |x| < 1.44e-2/log(2)
                cold_path();
                if ax < 0x3bac1405 {
                    // |x| < 3.64e-3/log(2)
                    cold_path();
                    if ax < 0x3a358876 {
                        // |x| < 4.8e-4/log(2)
                        cold_path();
                        if ax < 0x37d32ef6 {
                            // |x| < 1.745e-5/log(2)
                            cold_path();
                            if ax < 0x331fdd82 {
                                // |x| < 2.58e-8/log(2)
                                cold_path();
                                if ax < 0x2538aa3b {
                                    // |x| < 0x1.715476p-53
                                    cold_path();
                                    r = hf64!("0x1.62e42fefa39efp-1");
                                } else {
                                    r = hf64!("0x1.62e42fefa39fp-1")
                                        + z * hf64!("0x1.ebfbdff82c58fp-3");
                                }
                            } else {
                                if ux == 0xb3d8_5005 {
                                    cold_path();
                                    return (-hf64!("0x1.2bdf76p-24") - hf64!("0x1.8p-77"))
                                        .cast_lossy();
                                }

                                if ux == 0x3338_428d {
                                    cold_path();
                                    return (hf64!("0x1.fee08ap-26") + hf64!("0x1p-80"))
                                        .cast_lossy();
                                }

                                const C: [f64; 3] = [
                                    hf64!("0x1.62e42fefa39efp-1"),
                                    hf64!("0x1.ebfbdff8548fdp-3"),
                                    hf64!("0x1.c6b08d704a06dp-5"),
                                ];

                                r = C[0] + z * (C[1] + z * C[2]);
                            }
                        } else {
                            if ux == 0x388bca4f {
                                cold_path();
                                return (hf64!("0x1.839702p-15") - hf64!("0x1.8p-68")).cast_lossy();
                            }

                            const C: [f64; 4] = [
                                hf64!("0x1.62e42fefa39efp-1"),
                                hf64!("0x1.ebfbdff82c58fp-3"),
                                hf64!("0x1.c6b08dc82b347p-5"),
                                hf64!("0x1.3b2ab6fbad172p-7"),
                            ];
                            r = (C[0] + z * C[1]) + z2 * (C[2] + z * C[3]);
                        }
                    } else {
                        const C: [f64; 5] = [
                            hf64!("0x1.62e42fefa39efp-1"),
                            hf64!("0x1.ebfbdff82c068p-3"),
                            hf64!("0x1.c6b08d704a6dcp-5"),
                            hf64!("0x1.3b2ac262c3eedp-7"),
                            hf64!("0x1.5d87fe7af779ap-10"),
                        ];
                        r = (C[0] + z * C[1]) + z2 * (C[2] + z * (C[3] + z * C[4]));
                    }
                } else {
                    const C: [f64; 6] = [
                        hf64!("0x1.62e42fefa39fp-1"),
                        hf64!("0x1.ebfbdff82c58dp-3"),
                        hf64!("0x1.c6b08d7011d13p-5"),
                        hf64!("0x1.3b2ab6fbd267dp-7"),
                        hf64!("0x1.5d88a81cea49ep-10"),
                        hf64!("0x1.430912ea9b963p-13"),
                    ];
                    r = (C[0] + z * C[1]) + z2 * ((C[2] + z * C[3]) + z2 * (C[4] + z * C[5]));
                }
            } else {
                const C: [f64; 7] = [
                    hf64!("0x1.62e42fefa39efp-1"),
                    hf64!("0x1.ebfbdff82c639p-3"),
                    hf64!("0x1.c6b08d7049f1cp-5"),
                    hf64!("0x1.3b2ab6f5243bdp-7"),
                    hf64!("0x1.5d87fe80a9e6cp-10"),
                    hf64!("0x1.430d0b9257fa8p-13"),
                    hf64!("0x1.ffcbfc4cf0952p-17"),
                ];
                r = (C[0] + z * C[1])
                    + z2 * ((C[2] + z * C[3]) + z2 * (C[4] + z * (C[5] + z * C[6])));
            }
        } else {
            const C: [f64; 8] = [
                hf64!("0x1.62e42fefa39efp-1"),
                hf64!("0x1.ebfbdff82c591p-3"),
                hf64!("0x1.c6b08d704cf6bp-5"),
                hf64!("0x1.3b2ab6fba00cep-7"),
                hf64!("0x1.5d87fdfdaadb4p-10"),
                hf64!("0x1.4309137333066p-13"),
                hf64!("0x1.ffe5e90daf7ddp-17"),
                hf64!("0x1.62c0220eed731p-20"),
            ];

            r = ((C[0] + z * C[1]) + z2 * (C[2] + z * C[3]))
                + (z2 * z2) * ((C[4] + z * C[5]) + z2 * (C[6] + z * C[7]));
        }

        r *= z;
        r.cast_lossy()
    } else {
        const C: [f64; 6] = [
            hf64!("0x1.62e42fefa398bp-5"),
            hf64!("0x1.ebfbdff84555ap-11"),
            hf64!("0x1.c6b08d4ad86d3p-17"),
            hf64!("0x1.3b2ad1b1716a2p-23"),
            hf64!("0x1.5d7472718ce9dp-30"),
            hf64!("0x1.4a1d7f457ac56p-37"),
        ];

        const TB: [f64; 16] = [
            hf64!("0x1p+0"),
            hf64!("0x1.0b5586cf9890fp+0"),
            hf64!("0x1.172b83c7d517bp+0"),
            hf64!("0x1.2387a6e756238p+0"),
            hf64!("0x1.306fe0a31b715p+0"),
            hf64!("0x1.3dea64c123422p+0"),
            hf64!("0x1.4bfdad5362a27p+0"),
            hf64!("0x1.5ab07dd485429p+0"),
            hf64!("0x1.6a09e667f3bcdp+0"),
            hf64!("0x1.7a11473eb0187p+0"),
            hf64!("0x1.8ace5422aa0dap+0"),
            hf64!("0x1.9c49182a3f09p+0"),
            hf64!("0x1.ae89f995ad3adp+0"),
            hf64!("0x1.c199bdd85529cp+0"),
            hf64!("0x1.d5818dcfba487p+0"),
            hf64!("0x1.ea4afa2a490dap+0"),
        ];

        let a = 16.0 * z;
        let ia = Float::floor(a);
        let h = a - ia;
        let h2 = h * h;

        let i: i64 = ia.cast();
        let j = i & 0xf;
        let mut e = i - j;
        e >>= 4;

        let mut s = TB[usize::cast_from(j)];
        let su = f64::from_bits((e + 0x3ff).unsigned() << 52);
        s *= su;

        let mut c0 = C[0] + h * C[1];
        let c2 = C[2] + h * C[3];
        let c4 = C[4] + h * C[5];

        c0 += h2 * (c2 + h2 * c4);
        let w = s * h;
        ((s - 1.0) + w * c0).cast_lossy()
    }
}

#[cfg(test)]
mod tests {
    use super::exp2m1f;
    use crate::support::Float;

    #[test]
    fn known_results() {
        for (x, expected) in [
            (f32::ZERO, f32::ZERO),
            (f32::NEG_ZERO, f32::NEG_ZERO),
            (f32::ONE, f32::ONE),
            (f32::NEG_ONE, -0.5),
            (128.0, f32::INFINITY),
            (-25.0, f32::NEG_ONE),
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::NEG_ONE),
        ] {
            let result = exp2m1f(x);
            assert_biteq!(result, expected, "x = {x:?}");
        }

        for x in [f32::NAN, f32::NEG_NAN, f32::SNAN, f32::NEG_SNAN] {
            // NaN operations can change the sign or payload, including canonicalization on RISC-V.
            assert!(exp2m1f(x).is_nan(), "x = {x:?}");
        }
    }
}
