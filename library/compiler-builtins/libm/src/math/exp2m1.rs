/* SPDX-License-Identifier: MIT */
/* origin: core-math/src/binary64/exp2m1/exp2m1.c
 * Copyright (c) 2022-2025 Paul Zimmermann, Tom Hubrecht and Claude-Pierre Jeannerod
 * Ported to Rust in 2026, malezjaa
 * Approximate CORE-MATH commit: ac1993d82a2f83eadc045cb587fa8428c9ebaa16
 */

use super::ldexp;
use super::support::{CastFrom, CastInto, Float, Int, cold_path};

const LN2H: f64 = hf64!("0x1.62e42fefa39efp-1");
const LN2L: f64 = hf64!("0x1.abc9e3b39803fp-56");

/// Base-2 exponential minus one (f64)
///
/// Calculate `2^x - 1`, that is, 2 raised to the power `x` minus `1`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn exp2m1(x: f64) -> f64 {
    cr_exp2m1(x)
}

fn cr_exp2m1(x: f64) -> f64 {
    let ux = x.to_bits();
    let ax = ux & 0x7fffffffffffffff;

    if ux >= 0xc04b000000000000 {
        cold_path();

        // x = -NaN or x <= -54
        if (ux >> 52) == 0xfff {
            // -NaN or -Inf
            return if ux > 0xfff0000000000000 { x + x } else { -1.0 };
        }

        // for x <= -54, exp2m1(x) rounds to -1 to nearest
        return -1.0 + hf64!("0x1p-54");
    } else if ax >= 0x4090000000000000 {
        cold_path();
        // x = +NaN or x >= 1024
        if (ux >> 52) == 0x7ff {
            // +NaN
            return x + x;
        }

        // for x >= 1024, exp2m1(x) rounds to +Inf to nearest,
        // but for RNDZ/RNDD, we should have no overflow for x=1024

        return hf64!("0x1.fffffffffffffp+1023") + x * hf64!("0x1.fffffffffffffp+960");

    // |x| <= 0x1.0527dbd87e24dp-51
    } else if ax <= 0x3cc0527dbd87e24d
    /* then the second term of the Taylor expansion of 2^x-1 at x=0 is
    smaller in absolute value than 1/2 ulp(first term):
    log(2)*x + log(2)^2*x^2/2 + ... */
    {
        // we use special code when log(2)*|x| is very small, in which case
        // the double-double approximation hi+lo has its lower part lo
        // "truncated"
        // |x| <= 2^-104
        if ax <= 0x3970000000000000 {
            // special case for 0
            if x == 0.0 {
                return x;
            }

            // scale x by 2^106
            let x = x * hf64!("0x1p106");
            let (hi, lo) = a_mul(LN2H, x);
            let lo = LN2L.fma(x, lo);
            let h2 = hi + lo; // round to 53-bit precision
            // scale back, hoping to avoid double rounding
            let h2 = h2 * hf64!("0x1p-106");

            // now subtract back h2 * 2^106 from hi to get the correction term
            // add lo
            let hi = (-h2).fma(hf64!("0x1p106"), hi) + lo;

            // add h2 + hi * 2^-106. Warning: when hi=0, 2^-106*h2 might be exact,
            // thus no underflow will be raised. We have underflow for
            // 0 < x <= 0x1.71547652b82fep-1022 for RNDZ, and for
            // 0 < x <= 0x1.71547652b82fdp-1022 for RNDN/RNDU.
            let res = hi.fma(hf64!("0x1p-106"), h2);

            if ax <= 0x171547652b82fd || res.abs() < hf64!("0x1p-1022") {
                force_eval!(res * res); // underflow
            }

            return res;

        // 2^-104 < |x| <= 0x1.0527dbd87e24dp-51
        } else {
            // The following exceptional cases have at least 51 identical bits after
            // the round bit, thus are hard to correctly round with double-double
            // arithmetic. They should be sorted by increasing values of the first
            // entry (x).
            #[rustfmt::skip]
            static EXCEPTIONS_TABLE: [(f64, f64, f64); 56] = [
                (hf64!("-0x1.a16826a8e825dp-56"), hf64!("-0x1.21530a306cc85p-56"), hf64!("-0x1.38ac4a67cep-161")),
                (hf64!("-0x1.8c525b64ed08ep-59"), hf64!("-0x1.12b592f889516p-59"), hf64!("-0x1.7b71d1eep-169")),
                (hf64!("-0x1.0ede4c1293a9bp-59"), hf64!("-0x1.7780d5e5cf5c5p-60"), hf64!("-0x1.6b820f41efd9p-166")),
                (hf64!("-0x1.bacdbd3005cd7p-60"), hf64!("-0x1.32ed98e196cf5p-60"), hf64!("-0x1.9b32a24b8p-166")),
                (hf64!("-0x1.0481d96a2dfcap-61"), hf64!("-0x1.6923c31228cd3p-62"), hf64!("-0x1.59432bd8e301p-169")),
                (hf64!("-0x1.103fc46963aafp-62"), hf64!("-0x1.796ad95f38488p-63"), hf64!("-0x1.efd4112076edap-170")),
                (hf64!("-0x1.f1bc3ef3e6f36p-65"), hf64!("-0x1.5900fbf46981dp-65"), hf64!("-0x1.fffffffffffffp-119")),
                (hf64!("-0x1.d8bedc057858cp-65"), hf64!("-0x1.47aea7608c02bp-65"), hf64!("-0x1.ef8a5d5p-172")),
                (hf64!("-0x1.986d43391ffdp-65"), hf64!("-0x1.1b19925f9fc06p-65"), hf64!("-0x1.1aada6205370bp-170")),
                (hf64!("-0x1.6a480c34c7c5bp-70"), hf64!("-0x1.f63a8ce2364f3p-71"), hf64!("-0x1.11e154c338558p-177")),
                (hf64!("-0x1.03eda6663a4b2p-70"), hf64!("-0x1.6856506d8234ap-71"), hf64!("-0x1.f48368854ffd7p-178")),
                (hf64!("-0x1.ec44ae4bc644p-74"), hf64!("-0x1.5536e12eb7335p-74"), hf64!("-0x1.fb8acp-181")),
                (hf64!("-0x1.af82b29eef2ecp-74"), hf64!("-0x1.2b19ae19e4f3ep-74"), hf64!("-0x1.730d414c3032cp-180")),
                (hf64!("-0x1.0e8e362e16fe6p-74"), hf64!("-0x1.7711d03d5c7bdp-75"), hf64!("-0x1.63020ffb5bda3p-181")),
                (hf64!("-0x1.5754d81696c42p-76"), hf64!("-0x1.dbf55aa9a7eb9p-77"), hf64!("-0x1.c81798a2dea9ep-184")),
                (hf64!("-0x1.e2b5692bd0c53p-80"), hf64!("-0x1.4e968fb1b5f41p-80"), hf64!("0x1.fffffffffffffp-134")),
                (hf64!("-0x1.0042796d534fdp-81"), hf64!("-0x1.6340571968b67p-82"), hf64!("-0x1.91fa30638e3b8p-188")),
                (hf64!("-0x1.1e3a6eaa49c6ep-84"), hf64!("-0x1.8ccbeeaab37e4p-85"), hf64!("0x1p-138")),
                (hf64!("-0x1.19f9b6ddcc3e9p-84"), hf64!("-0x1.86e6a6125ee5fp-85"), hf64!("-0x1.d80acb1fd9a48p-191")),
                (hf64!("-0x1.9b219a1974289p-86"), hf64!("-0x1.1cf977003bdd6p-86"), hf64!("-0x1.2a8a589db0f16p-194")),
                (hf64!("-0x1.b102d7663223fp-87"), hf64!("-0x1.2c23f2bc02277p-87"), hf64!("-0x1.1304b8a0f41cdp-194")),
                (hf64!("-0x1.3db1f733be456p-87"), hf64!("-0x1.b86b45d2c7c74p-88"), hf64!("-0x1.65075b3baa854p-194")),
                (hf64!("-0x1.712c6b2a267f5p-90"), hf64!("-0x1.ffc87ce076dfap-91"), hf64!("-0x1.922d1954bd733p-196")),
                (hf64!("-0x1.8bf6e9484dd46p-91"), hf64!("-0x1.127630515eb89p-91"), hf64!("-0x1.aa2e8ae4b621cp-198")),
                (hf64!("-0x1.56fe8536a47e9p-91"), hf64!("-0x1.db7daf1e016dfp-92"), hf64!("-0x1.5a5357cd9fecap-199")),
                (hf64!("-0x1.4abb72eb058bep-91"), hf64!("-0x1.ca7e01c959788p-92"), hf64!("-0x1.9c974667cda43p-198")),
                (hf64!("-0x1.cf23f6232620ep-94"), hf64!("-0x1.4106468e7b671p-94"), hf64!("-0x1.a1de1e34f2642p-200")),
                (hf64!("-0x1.0c197520b26f4p-94"), hf64!("-0x1.73aa2cd72b7ccp-95"), hf64!("-0x1.1p-148")),
                (hf64!("-0x1.83266a65e67b3p-95"), hf64!("-0x1.0c5a1aeb10a7cp-95"), hf64!("-0x1.5d790953d37ccp-202")),
                (hf64!("-0x1.e48acc7d41976p-97"), hf64!("-0x1.4fdbea8f31897p-97"), hf64!("-0x1.889c4c9736da4p-202")),
                (hf64!("-0x1.241e7c1327b2ap-97"), hf64!("-0x1.94f6896c09e68p-98"), hf64!("-0x1.ad2fd62d965b7p-203")),
                (hf64!("-0x1.c48ddf5beba98p-98"), hf64!("-0x1.39afc8fadbb98p-98"), hf64!("-0x1.37656827f3549p-203")),
                (hf64!("-0x1.1d65d5fc31246p-98"), hf64!("-0x1.8ba5360a2b553p-99"), hf64!("-0x1.0bee24969cff9p-204")),
                (hf64!("0x1.8a5dd21ef35afp-104"), hf64!("0x1.115aa0fb29a7p-104"), hf64!("-0x1.1p-157")),
                (hf64!("0x1.51cc163375e5cp-100"), hf64!("0x1.d4494fb79c5f9p-101"), hf64!("0x1.235a318115a7dp-210")),
                (hf64!("0x1.086cbb0e900cfp-96"), hf64!("0x1.6e920d043905ep-97"), hf64!("0x1.0000000000001p-150")),
                (hf64!("0x1.f8d2954ab444fp-91"), hf64!("0x1.5dea9642be3ccp-91"), hf64!("0x1.109cd4f731735p-199")),
                (hf64!("0x1.8b571f80771bep-90"), hf64!("0x1.12076e976275dp-90"), hf64!("-0x1.fffffffffffffp-144")),
                (hf64!("0x1.453f5d718374ap-88"), hf64!("0x1.c2e3888d4911bp-89"), hf64!("0x1.fffffffffffffp-143")),
                (hf64!("0x1.101c8b0d7df57p-85"), hf64!("0x1.793a04a8764ap-86"), hf64!("0x1.95385b3628535p-194")),
                (hf64!("0x1.99ec5e10fdc8bp-83"), hf64!("0x1.1c231eacb2814p-83"), hf64!("0x1.f82d478ebc7fap-192")),
                (hf64!("0x1.530a89e8c2834p-79"), hf64!("0x1.d602c792ff217p-80"), hf64!("0x1.4d765bf788c32p-186")),
                (hf64!("0x1.6102ff22f3431p-78"), hf64!("0x1.e960cd938fc71p-79"), hf64!("0x1.fffffffffffffp-133")),
                (hf64!("0x1.d7430e6a40aap-78"), hf64!("0x1.46a764f31c407p-78"), hf64!("0x1.fffffffffffffp-132")),
                (hf64!("0x1.31677c77b4d02p-62"), hf64!("0x1.a7615378454e9p-63"), hf64!("-0x1.fffffffffffffp-117")),
                (hf64!("0x1.760aba9c35ed1p-62"), hf64!("0x1.03441ed22884fp-62"), hf64!("-0x1.fffffffffffffp-116")),
                (hf64!("0x1.5f3addb5548c6p-60"), hf64!("0x1.e6e878c6cb8f4p-61"), hf64!("0x1.46dbd6dc7b9eep-168")),
                (hf64!("0x1.9a8401f6d9784p-60"), hf64!("0x1.1c8c3a93ce469p-60"), hf64!("-0x1.fffffffffffffp-114")),
                (hf64!("0x1.7382bd647eca8p-59"), hf64!("0x1.0182f7f3350e1p-59"), hf64!("0x1.5902345c6d87ep-166")),
                (hf64!("0x1.8ddd5485a342fp-56"), hf64!("0x1.13c75940129cbp-56"), hf64!("0x1.365292846d16p-164")),
                (hf64!("0x1.2736f975d3b57p-55"), hf64!("0x1.994129328c805p-56"), hf64!("-0x1.fffffffffffffp-110")),
                (hf64!("0x1.690ed9e3dd31p-55"), hf64!("0x1.f4885e22dc71bp-56"), hf64!("-0x1.fffffffffffffp-110")),
                (hf64!("0x1.e704310d30238p-55"), hf64!("0x1.5192f360cac0ap-55"), hf64!("0x1.30562032a2c95p-161")),
                (hf64!("0x1.47bc52cdc6bc1p-53"), hf64!("0x1.c6568b98a9929p-54"), hf64!("0x1.3977497829482p-159")),
                (hf64!("0x1.d622a201c030bp-52"), hf64!("0x1.45df797393f11p-52"), hf64!("-0x1.cdcde906092cfp-159")),
                (hf64!("0x1.02f371449097fp-51"), hf64!("0x1.66fb73eec9971p-52"), hf64!("0x1.8d182ad6f54d8p-157")),
            ];

            if let Some(result) = lookup_exception(&EXCEPTIONS_TABLE, x) {
                return result;
            }

            // the 2nd term of the Taylor expansion of 2^x-1 at x=0 is
            // log(2)^2/2*x^2
            const C2: f64 = hf64!("0x1.ebfbdff82c58fp-3"); // log(2)^2/2

            let (hi, lo) = a_mul(LN2H, x);
            let lo = LN2L.fma(x, lo);

            // hi+lo approximates the first term x*log(2)
            // we add C2*x^2 last, so that in case there is a cancellation in
            // LN2L*x+lo, it will contribute more bits

            let lo = lo + (C2 * x * x);
            return hi + lo;
        }
    }

    // now -54 < x < -0x1.0527dbd87e24dp-51
    // or 0x1.0527dbd87e24dp-51 < x < 1024

    // 2^x-1 is exact for x integer, -53 <= x <= 53
    if ux << 17 == 0 {
        cold_path();

        let i = Float::floor(x);
        if x == i && (-53.0..=53.0).contains(&i) {
            if i >= 0.0 {
                return ((1u64 << i64::cast_from(i)) - 1).cast();
            } else {
                return -1.0 + ldexp(1.0, i.cast());
            }
        }
    }

    let (hi, lo, err) = exp2m1_fast(x, ax <= 0x3fc0000000000000);
    let left = hi + (lo - err);
    let right = hi + (lo + err);

    if left == right {
        return left;
    }

    cold_path();
    exp2m1_accurate(x)
}

/// Given `-54 < x < -0x1.0527dbd87e24dp-51` or `0x1.0527dbd87e24dp-51 < x < 1024`, put in `hi + lo`
/// a double-double approximation of `exp2m1(x)`, and return the maximal corresponding
/// absolute error.
/// The input `tiny` is true iff `|x| <= 0.125`.
#[inline]
fn exp2m1_fast(x: f64, tiny: bool) -> (f64, f64, f64) {
    // |x| <= 0.125
    if tiny {
        return exp2m1_fast_tiny(x);
    }

    // now -54 < x < -0.125 or 0.125 < x < 1024: we approximate exp(x*log(2))
    // and subtract 1
    let (hi, lo) = s_mul(x, LN2H, LN2L);

    // s_mul (hi, lo, x, LN2H, LN2L) decomposes into:
    // a_mul (hi, t, x, LN2H)
    // lo += fma (x, LN2L, t)
    // The a_mul() call is exact, and the error of the fma() is bounded by
    // ulp(lo).
    // We have |t| <= ulp(hi) <= ulp(LN2H*1024) = 2^-43,
    // |t+x*LN2L| <= 2^-43 * 1024*LN2L < 2^-42.7,
    // thus |lo| <= |t| + |x*LN2L| + ulp(t+x*LN2L)
    //           <= 2^-42.7 + 2^-95 <= 2^-42.6, and ulp(lo) <= 2^-95.
    // Thus:
    // |hi + lo - x*log(2)| <= |hi + lo - x*(LN2H+LN2L)| + |x|*|LN2H+LN2L-log(2)|
    //                      <= 2^-95 + 1024*2^-110.4 < 2^-94.9

    let (hi, lo) = exp_1(hi, lo);
    // hi_out + lo_out = exp(hi + lo) * (1 + eps) with |eps| < 2^-74.139
    // = exp(x*log(2) + eps0) * (1 + eps)
    // with |eps0| < 2^-94.9 and |eps| < 2^-74.139
    // = 2^x * (1 + eps1) with |eps1| < 2^-74.138

    // implies hi >= 1 and the fast_two_sum pre-condition holds
    let (hi, u) = if x >= 0.0 {
        fast_two_sum(hi, -1.0)
    } else {
        fast_two_sum(-1.0, hi) // x < 0 thus hi <= 1
    };
    let lo = lo + u;

    // The error in the above fast_two_sum is bounded by 2^-105*|hi|,
    // with the new value of hi, thus the total absolute error is bounded
    // by eps1*|hi_in|+2^-105*|hi|.
    // Relatively to hi this yields eps1*|hi_in/hi| + 2^-105, where the maximum
    // of |hi_in/hi| is obtained for x near -0.125, with |2^x/(2^x-1)| < 11.05.
    // We get a relative error bound of 2^-74.138*11.05 + 2^-105 < 2^-70.67.
    let err = hf64!("0x1.42p-71") * hi; // 2^-70.67 < 0x1.42p-71
    (hi, lo, err)
}

#[cold]
#[inline]
fn exp2m1_accurate(x: f64) -> f64 {
    let ux = x.to_bits();
    let ax = ux & 0x7fffffffffffffff;

    // |x| <= 0.125
    if ax <= 0x3fc0000000000000 {
        return exp2m1_accurate_tiny(x);
    }

    // now -54 < x < -0.125 or 0.125 < x < 1024: we approximate exp(x*log(2))
    // and subtract 1
    // The following table should be sorted by increasing values of the first
    // entry (x).
    #[rustfmt::skip]
    static EXCEPTIONS_TABLE: [(f64, f64, f64); 93] = [
        (-hf64!("0x1.da22611253866p+0"), -hf64!("0x1.722e3006b8e0bp-1"), -hf64!("0x1.cb1ddc7fbf64ep-109")),
        (-hf64!("0x1.9e5bc7d550bb5p+0"), -hf64!("0x1.5943df965427p-1"), hf64!("0x1.f00a26c16d035p-107")),
        (-hf64!("0x1.e511f8c5f829fp-1"), -hf64!("0x1.ecfd3f665583fp-2"), hf64!("0x1.cd0996b2d6842p-106")),
        (-hf64!("0x1.cef4c143b5adfp-1"), -hf64!("0x1.dcd9f4d61d31fp-2"), hf64!("0x1.88f636e171893p-109")),
        (-hf64!("0x1.b444c224a70ccp-1"), -hf64!("0x1.c8b8c01ae382cp-2"), -hf64!("0x1.cb1ddc7fbf64ep-108")),
        (-hf64!("0x1.9de261c7c8623p-1"), -hf64!("0x1.b7448eda85236p-2"), hf64!("0x1.e2651db687d79p-106")),
        (-hf64!("0x1.750291d22fdap-1"), -hf64!("0x1.95ffcf1509869p-2"), hf64!("0x1.2aabef02ae4cbp-103")),
        (-hf64!("0x1.3cb78faaa176ap-1"), -hf64!("0x1.650f7e59509cp-2"), hf64!("0x1.f00a26c16d035p-106")),
        (-hf64!("0x1.304447dcfa48ap-1"), -hf64!("0x1.59b94aff7c2b5p-2"), hf64!("0x1.ca6392b99ad78p-106")),
        (-hf64!("0x1.e242801b45d0dp-2"), -hf64!("0x1.1d32c9d218361p-2"), hf64!("0x1.03dd41d134594p-109")),
        (-hf64!("0x1.d2b12277c95d4p-2"), -hf64!("0x1.155f2f4b7a739p-2"), hf64!("0x1.08b48f6466755p-106")),
        (-hf64!("0x1.a4ec34efdd3e1p-2"), -hf64!("0x1.fbc1acbd1558ep-3"), hf64!("0x1.be847d667711dp-105")),
        (-hf64!("0x1.8827da1dbf35ep-2"), -hf64!("0x1.dd7867fe8d091p-3"), hf64!("0x1.ffffffffffff8p-57")),
        (-hf64!("0x1.75cfa7af70fep-2"), -hf64!("0x1.c9d8c829dad99p-3"), -hf64!("0x1.7e233fcc3fd44p-105")),
        (-hf64!("0x1.6d729ff937eadp-2"), -hf64!("0x1.c0d1b0bcb29d6p-3"), hf64!("0x1.7f0f685911fccp-105")),
        (-hf64!("0x1.6b3801c55815ap-2"), -hf64!("0x1.be67897b4ab5ep-3"), hf64!("0x1.636be28bf2c88p-106")),
        (-hf64!("0x1.6822ccd5b876bp-2"), -hf64!("0x1.bb0f0bff2db03p-3"), hf64!("0x1.4d6bbe2b72be3p-106")),
        (-hf64!("0x1.5dbf023667d22p-2"), -hf64!("0x1.afbb4c7215b5bp-3"), hf64!("0x1.3bbc634b006b1p-105")),
        (-hf64!("0x1.5a0fba59f338fp-2"), -hf64!("0x1.abb2028f6491cp-3"), hf64!("0x1.0a9c5a206828ep-104")),
        (-hf64!("0x1.540658c7eedcap-2"), -hf64!("0x1.a50f96bbf778cp-3"), hf64!("0x1.f14fe3c42820cp-105")),
        (-hf64!("0x1.31c2bbece259bp-2"), -hf64!("0x1.7ee2b68e61af7p-3"), -hf64!("0x1.7852d3ddee861p-107")),
        (-hf64!("0x1.253dd048bf2e7p-2"), -hf64!("0x1.70b709e65877ep-3"), hf64!("0x1.3625f20d5b638p-104")),
        (-hf64!("0x1.1249283f53506p-2"), -hf64!("0x1.5b076e9ede20bp-3"), hf64!("0x1.14d2b0b9298b2p-105")),
        (-hf64!("0x1.0f2086c7b8f83p-2"), -hf64!("0x1.57636305fd889p-3"), hf64!("0x1.9442185db9d73p-104")),
        (-hf64!("0x1.ff6ff72ac299dp-3"), -hf64!("0x1.458423bb5b025p-3"), hf64!("0x1.140aa7335a14p-104")),
        (-hf64!("0x1.fe89353e31cbfp-3"), -hf64!("0x1.44fd97cc9857p-3"), -hf64!("0x1.a549adf007feap-109")),
        (-hf64!("0x1.ec7298061ebfap-3"), -hf64!("0x1.3a68f321de8abp-3"), hf64!("0x1.7f700400ca84fp-103")),
        (-hf64!("0x1.d4854d9f87fcap-3"), -hf64!("0x1.2c506fcda97fcp-3"), -hf64!("0x1.a4fedf7b3ca3fp-109")),
        (-hf64!("0x1.a81224ff9a5cfp-3"), -hf64!("0x1.11d2b7526eae5p-3"), hf64!("0x1.6fb9bdbb725b2p-106")),
        (-hf64!("0x1.93314fd04393cp-3"), -hf64!("0x1.053dee1b48ebdp-3"), hf64!("0x1.2c15fbfa775b3p-104")),
        (-hf64!("0x1.8cd71c092c54dp-3"), -hf64!("0x1.0165759fe661fp-3"), hf64!("0x1.395807142ed1ep-104")),
        (-hf64!("0x1.747fb97a9be61p-3"), -hf64!("0x1.e52ae5003c9dbp-4"), -hf64!("0x1.dbf62ab30e6cp-106")),
        (-hf64!("0x1.64b0e525bc004p-3"), -hf64!("0x1.d1cc0e1d08fa9p-4"), hf64!("0x1.69442904d7698p-104")),
        (-hf64!("0x1.539353db1bdebp-3"), -hf64!("0x1.bcb50aacb6649p-4"), -hf64!("0x1.a8b1eddf109b5p-107")),
        (-hf64!("0x1.4faff91bda7ddp-3"), -hf64!("0x1.b7e630a3fab2ap-4"), hf64!("0x1.47706b5ad802ap-105")),
        (-hf64!("0x1.4c2bec34d5837p-3"), -hf64!("0x1.b38bcbad2b407p-4"), -hf64!("0x1.b4e03df36240cp-108")),
        (-hf64!("0x1.439bb08833472p-3"), -hf64!("0x1.a8ebef4998a79p-4"), -hf64!("0x1.2c3a2b30f99cfp-105")),
        (-hf64!("0x1.418f83945849ap-3"), -hf64!("0x1.a6606e818c2e3p-4"), hf64!("0x1.145b99cfb763cp-104")),
        (-hf64!("0x1.38417d7db5a2fp-3"), -hf64!("0x1.9aca18a87c03ap-4"), hf64!("0x1.76a10886fcc65p-105")),
        (-hf64!("0x1.3041fe0fc550bp-3"), -hf64!("0x1.90ccdeae8d3ddp-4"), hf64!("0x1.f202cb751932p-104")),
        (-hf64!("0x1.236e786c083dbp-3"), -hf64!("0x1.80b96ddbdb1b3p-4"), hf64!("0x1.ec381ad5e1718p-105")),
        (-hf64!("0x1.06c0eba373a18p-3"), -hf64!("0x1.5c86ed4f55301p-4"), hf64!("0x1.fffffffffffe3p-58")),
        (-hf64!("0x1.06559d8a9ff1ap-3"), -hf64!("0x1.5bfed18cc8ff3p-4"), hf64!("0x1.87a53f06d41bdp-105")),
        (-hf64!("0x1.0006ec0817742p-3"), -hf64!("0x1.53fc5e2bd9f6cp-4"), -hf64!("0x1.6fcda9953133bp-107")),
        (hf64!("0x1.06d6b01577482p-3"), hf64!("0x1.7d11edfaac429p-4"), hf64!("0x1.ffffffffffffbp-58")),
        (hf64!("0x1.079724fbb3d86p-3"), hf64!("0x1.7e35970e11732p-4"), -hf64!("0x1.b3086fea6e73cp-107")),
        (hf64!("0x1.0e2f828c93192p-3"), hf64!("0x1.88376cd2f4076p-4"), -hf64!("0x1.0bc0000d21ae1p-106")),
        (hf64!("0x1.111c072cd8f6ep-3"), hf64!("0x1.8ca9045ba3e95p-4"), hf64!("0x1.80f80d8a8722ap-105")),
        (hf64!("0x1.165c1c6243beep-3"), hf64!("0x1.94a6924c0b159p-4"), hf64!("0x1.ffffffffffff9p-58")),
        (hf64!("0x1.182f8cb329d34p-3"), hf64!("0x1.976ed12501a66p-4"), -hf64!("0x1.b0e5142b2ec14p-105")),
        (hf64!("0x1.1a8801599f756p-3"), hf64!("0x1.9b02639918dcdp-4"), hf64!("0x1.736bd9c878fbap-107")),
        (hf64!("0x1.1bb1b31cf8c98p-3"), hf64!("0x1.9cc894e900c64p-4"), hf64!("0x1.218cf2d9f3926p-104")),
        (hf64!("0x1.2ba99934336fcp-3"), hf64!("0x1.b5379b9a6155cp-4"), hf64!("0x1.68cdeb7f25f81p-106")),
        (hf64!("0x1.2dbe99fec947ap-3"), hf64!("0x1.b869aac963d94p-4"), hf64!("0x1.59f0621f5fc6ep-107")),
        (hf64!("0x1.2eecf76d63cdp-3"), hf64!("0x1.ba39ff28e3eap-4"), -hf64!("0x1.cb1ddc7fbf64ep-107")),
        (hf64!("0x1.45c447c5552ddp-3"), hf64!("0x1.dd721415a4a8dp-4"), -hf64!("0x1.e5111c4062fa9p-108")),
        (hf64!("0x1.502f26bc1aa18p-3"), hf64!("0x1.ed99608511a69p-4"), -hf64!("0x1.603952f77b6d4p-106")),
        (hf64!("0x1.5856bdae2f08fp-3"), hf64!("0x1.fa489ed69464fp-4"), hf64!("0x1.93f08a7946826p-104")),
        (hf64!("0x1.5f38bbc1a9944p-3"), hf64!("0x1.028230c4e5389p-3"), hf64!("0x1.313458d3f84a2p-105")),
        (hf64!("0x1.691b76c4b3c1bp-3"), hf64!("0x1.0a3d166eacab7p-3"), hf64!("0x1.9a98ee8a745ffp-104")),
        (hf64!("0x1.6c4175ea0c6e1p-3"), hf64!("0x1.0cb4b9c7701a4p-3"), hf64!("0x1.a6605f3be564ep-108")),
        (hf64!("0x1.6f6544ef96539p-3"), hf64!("0x1.0f2b51ddc90e1p-3"), hf64!("0x1.06e9b6d330daep-105")),
        (hf64!("0x1.7a3ab625d0dc4p-3"), hf64!("0x1.17b04242e48fep-3"), hf64!("0x1.60215b123bdbdp-108")),
        (hf64!("0x1.7d33f0271269bp-3"), hf64!("0x1.1a0842d5b11e4p-3"), -hf64!("0x1.5c29b4c0046f9p-106")),
        (hf64!("0x1.887678e0de774p-3"), hf64!("0x1.22edc495eb728p-3"), hf64!("0x1.e2651db687d79p-105")),
        (hf64!("0x1.8fe3adec251f6p-3"), hf64!("0x1.28d0ae0c0ac4fp-3"), hf64!("0x1.e1bce5201668bp-104")),
        (hf64!("0x1.926961243babap-3"), hf64!("0x1.2ad15442773c8p-3"), hf64!("0x1.112edd525e11cp-105")),
        (hf64!("0x1.a066bd757bf54p-3"), hf64!("0x1.35f4af72a0a25p-3"), -hf64!("0x1.59b2d13ebb9d3p-106")),
        (hf64!("0x1.b00f80708d157p-3"), hf64!("0x1.427c60cc0d217p-3"), hf64!("0x1.344b77789552ap-103")),
        (hf64!("0x1.b8ee8dcb0e797p-3"), hf64!("0x1.499d2a386003fp-3"), -hf64!("0x1.e5d43d3c21c2ap-105")),
        (hf64!("0x1.c3b0f79893bfcp-3"), hf64!("0x1.5249b249243b7p-3"), hf64!("0x1.1a9819f194524p-105")),
        (hf64!("0x1.d5ac74e107b39p-3"), hf64!("0x1.60db0007c78ffp-3"), -hf64!("0x1.a936bca36028fp-106")),
        (hf64!("0x1.e9532c87abc5bp-3"), hf64!("0x1.70e070f8d9706p-3"), -hf64!("0x1.0ab97f38bd529p-106")),
        (hf64!("0x1.16383a5fe20dp-2"), hf64!("0x1.a867756a9a8fep-3"), -hf64!("0x1.8ac73f07192e6p-106")),
        (hf64!("0x1.2a381dbbc8f07p-2"), hf64!("0x1.ca1a3ba1db0ebp-3"), -hf64!("0x1.7f5a08441b722p-105")),
        (hf64!("0x1.8690e0aabd12cp-2"), hf64!("0x1.35e1034d5ec8p-2"), hf64!("0x1.f00a26c16d035p-105")),
        (hf64!("0x1.a246f399a957bp-2"), hf64!("0x1.4f22bccd234f1p-2"), hf64!("0x1.ffffffffffff6p-56")),
        (hf64!("0x1.b808c857723e7p-2"), hf64!("0x1.634ce84a0e79dp-2"), hf64!("0x1.ffffffffffff4p-56")),
        (hf64!("0x1.bb2446a7b770fp-2"), hf64!("0x1.663466e4bb92bp-2"), -hf64!("0x1.a00681268d32cp-106")),
        (hf64!("0x1.c14d1ce57d7ebp-2"), hf64!("0x1.6bfacbdf4334ep-2"), hf64!("0x1.1cfc31c20bd19p-104")),
        (hf64!("0x1.f8432a3c6c453p-2"), hf64!("0x1.a096f253eca4bp-2"), hf64!("0x1.56c0f7fb6fa16p-106")),
        (hf64!("0x1.16a76ec41b516p-1"), hf64!("0x1.d541a1690b18ep-2"), hf64!("0x1.08b48f6466755p-105")),
        (hf64!("0x1.1e5f48d8ba05ap-1"), hf64!("0x1.e4f0e78affdcdp-2"), hf64!("0x1.ffffffffffffep-56")),
        (hf64!("0x1.4a63ff1d53f53p-1"), hf64!("0x1.20cc3b425aa51p-1"), hf64!("0x1.636be28bf2c88p-105")),
        (hf64!("0x1.d906bc1caeb3dp-1"), hf64!("0x1.cb5f21a856e8dp-1"), hf64!("0x1.c737d097094f7p-104")),
        (hf64!("0x1.25dd9eedac79ap+0"), hf64!("0x1.37473fe51c7d4p+0"), -hf64!("0x1.cb1ddc7fbf64ep-106")),
        (hf64!("0x1.61a4382aaf44bp+0"), hf64!("0x1.9af081a6af64p+0"), hf64!("0x1.f00a26c16d035p-104")),
        (hf64!("0x1.8b53b7620da8bp+0"), hf64!("0x1.eaa0d0b4858c7p+0"), hf64!("0x1.08b48f6466755p-104")),
        (hf64!("0x1.12eecf76d63cdp+1"), hf64!("0x1.b7473fe51c7d4p+1"), -hf64!("0x1.cb1ddc7fbf64ep-105")),
        (hf64!("0x1.92eecf76d63cdp+1"), hf64!("0x1.f7473fe51c7d4p+2"), -hf64!("0x1.cb1ddc7fbf64ep-104")),
        (hf64!("0x1.ffef3f31766b7p+5"), hf64!("0x1.fd1ae8d0e6abdp+63"), -hf64!("0x1.5ea4d3ad23ccep-42")),
        (hf64!("0x1.586b62dc1e8c2p+6"), hf64!("0x1.134d37339fe93p+86"), -hf64!("0x1.1295610d3deffp-20")),
        (hf64!("0x1.6f33cdaf56d6p+6"), hf64!("0x1.bde76c2db52fbp+91"), hf64!("0x1.564254439c8a9p-15")),
    ];

    if let Some(result) = lookup_exception(&EXCEPTIONS_TABLE, x) {
        return result;
    }

    let (hi, lo) = exp_2(x);

    // implies hi >= 1 and the fast_two_sum pre-condition holds
    let (hi, u) = if x >= 0.0 {
        fast_two_sum(hi, -1.0)
    } else {
        fast_two_sum(-1.0, hi) // x < 0 thus hi <= 1
    };

    let lo = lo + u;

    // the error in the above fast_two_sum is bounded by 2^-105*|hi|,
    // with the new value of hi
    return hi + lo;
}

/// The following is a degree-10 polynomial generated by Sollya (file `exp2m1_fast.sollya`),
/// which approximates `exp2m1(x)` with relative error bounded by `2^-68.559` for `|x| <= 0.125`.
const P: [f64; 12] = [
    hf64!("0x1.62e42fefa39efp-1"),
    hf64!("0x1.abd1697afcaf8p-56"), // degree 1, P[0], P[1]
    hf64!("0x1.ebfbdff82c58fp-3"),
    -hf64!("0x1.5e5a1d09e1599p-57"), // degree 2, P[2], P[3]
    hf64!("0x1.c6b08d704a0bfp-5"),   // degree 3, P[4]
    hf64!("0x1.3b2ab6fba4e78p-7"),   // degree 4, P[5]
    hf64!("0x1.5d87fe78a84e6p-10"),  // degree 5, P[6]
    hf64!("0x1.430912f86a48p-13"),   // degree 6, P[7]
    hf64!("0x1.ffcbfbc1f2b36p-17"),  // degree 7, P[8]
    hf64!("0x1.62c0226c7f6d1p-20"),  // degree 8, P[9]
    hf64!("0x1.b539529819e63p-24"),  // degree 9, P[10]
    hf64!("0x1.e4d552bed5b9cp-28"),  // degree 10, P[11]
];

/// `|x| <= 0.125`, put in `hi + lo` a double-double approximation of `exp2m1(x)`, and return the
/// maximal corresponding absolute error. We also have `|x| > 0x1.0527dbd87e24dp-51`.
/// With `xmin=RR("0x1.0527dbd87e24dp-51",16)`, the routine
/// `exp2m1_fast_tiny_all(xmin,0.125,2^-65.73)` in `exp2m1.sage` returns
/// `1.63414352331297e-20 < 2^-65.73`, and `exp2m1_fast_tiny_all(-0.125,-xmin,2^-65.62)` returns
/// `1.76283772822891e-20 < 2^-65.62`, which proves the relative error is bounded by `2^-65.62`.
#[inline]
fn exp2m1_fast_tiny(x: f64) -> (f64, f64, f64) {
    // The maximal value of |c4*x^4/exp2m1(x)| over [-0.125,0.125]
    // is less than 2^-15.109, where c4 is the degree-4 coefficient,
    // thus we can compute the coefficients of degree 4 or higher
    // using double precision only.
    let x2 = x * x;
    let x4 = x2 * x2;
    let c8 = P[10].fma(x, P[9]); // degree 8
    let c6 = P[8].fma(x, P[7]); // degree 6
    let c4 = P[6].fma(x, P[5]); // degree 4
    let c8 = P[11].fma(x2, c8); // degree 8
    let c4 = c6.fma(x2, c4); // degree 4
    let c4 = c8.fma(x4, c4); // degree 4

    // multiply c4 by x and add P[4]
    let (hi, lo) = a_mul(c4, x);
    let (hi, lo) = fast_sum(P[4], hi, lo);

    // multiply (hi,lo) by x and add P[2]+P[3]
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(P[2], hi);
    let lo = lo + (t + P[3]);

    // multiply (hi,lo) by x and add P[0]+P[1]
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(P[0], hi);
    let lo = lo + (t + P[1]);

    // multiply (hi,lo) by x
    let (hi, lo) = s_mul(x, hi, lo);
    let err = hf64!("0x1.4ep-66") * hi; // 2^-65.62 < 0x1.4ep-66
    (hi, lo, err)
}

// The following is a degree-15 polynomial generated by Sollya
// (file exp2m1_accurate.sollya),
// which approximates exp2m1(x) with relative error bounded by 2^-107.666
// for |x| <= 0.125.
const Q: [f64; 22] = [
    hf64!("0x1.62e42fefa39efp-1"),
    hf64!("0x1.abc9e3b39804p-56"), // degree 1: Q[0], Q[1]
    hf64!("0x1.ebfbdff82c58fp-3"),
    -hf64!("0x1.5e43a53e44dcfp-57"), // degree 2: Q[2], Q[3]
    hf64!("0x1.c6b08d704a0cp-5"),
    -hf64!("0x1.d331627517168p-59"), // degree 3: Q[4], Q[5]
    hf64!("0x1.3b2ab6fba4e77p-7"),
    hf64!("0x1.4e65df0779f8cp-62"), // degree 4: Q[6], Q[7]
    hf64!("0x1.5d87fe78a6731p-10"),
    hf64!("0x1.0717fbf4bd05p-66"), // degree 5: Q[8], Q[9]
    hf64!("0x1.430912f86c787p-13"),
    hf64!("0x1.bd2bdec9bcd42p-67"), // degree 6: Q[10], Q[11]
    hf64!("0x1.ffcbfc588b0c7p-17"),
    -hf64!("0x1.e60aa6d5e4aa9p-71"), // degree 7: Q[12], Q[13]
    hf64!("0x1.62c0223a5c824p-20"),  // degree 8: Q[14]
    hf64!("0x1.b5253d395e7d4p-24"),  // degree 9: Q[15]
    hf64!("0x1.e4cf5158b916p-28"),   // degree 10: Q[16]
    hf64!("0x1.e8cac734c6058p-32"),  // degree 11: Q[17]
    hf64!("0x1.c3bd64f17199dp-36"),  // degree 12: Q[18]
    hf64!("0x1.8161a17e05651p-40"),  // degree 13: Q[19]
    hf64!("0x1.3150b3d792231p-44"),  // degree 14: Q[20]
    hf64!("0x1.c184260bfad7ep-49"),  // degree 15: Q[21]
];

/// Accurate path for `0x1.0527dbd87e24dp-51 < |x| <= 0.125`.
#[inline]
fn exp2m1_accurate_tiny(x: f64) -> f64 {
    // The following exceptional cases have at least 51 identical bits after
    // the round bit, thus are hard to correctly round with double-double
    // arithmetic. They should be sorted by increasing values of the first
    // entry (x).
    #[rustfmt::skip]
    static EXCEPTIONS_TABLE: [(f64, f64, f64); 59] = [
        (-hf64!("0x1.f6ec73d3948c3p-4"), -hf64!("0x1.4e2d8b0cead45p-4"), -hf64!("0x1.40f3d5244acffp-109")),
        (-hf64!("0x1.9b28778a9a8a3p-4"), -hf64!("0x1.134dec3a6324bp-4"), -hf64!("0x1.1acaaac7e1ff7p-110")),
        (-hf64!("0x1.6f94484e5e1fdp-5"), -hf64!("0x1.f5baee010ccc6p-6"), -hf64!("0x1.3298bcde4f9a8p-115")),
        (-hf64!("0x1.3918e8608bd5bp-8"), -hf64!("0x1.b153bf52832f9p-9"), -hf64!("0x1.fffffffffffffp-63")),
        (-hf64!("0x1.8474969f5eb14p-9"), -hf64!("0x1.0cfafc07a957bp-9"), -hf64!("0x1.fffffffffffffp-63")),
        (-hf64!("0x1.0867d8153350dp-9"), -hf64!("0x1.6e49b44e387f5p-10"), -hf64!("0x1.fffffffffffffp-64")),
        (-hf64!("0x1.bfb3efcdf2bc4p-10"), -hf64!("0x1.3623eff91de91p-10"), hf64!("0x1.cd506bd8d0439p-117")),
        (-hf64!("0x1.0c3ebbd1a501fp-11"), -hf64!("0x1.73ccf8ee62819p-12"), hf64!("0x1.3a3965a926c6dp-120")),
        (-hf64!("0x1.3c7971b0ee205p-14"), -hf64!("0x1.b6b716c4bb87cp-15"), hf64!("0x1.d7150e84d973dp-121")),
        (-hf64!("0x1.3dbf41403c0b2p-15"), -hf64!("0x1.b87c37192e4f7p-16"), hf64!("0x1.098861b427b13p-123")),
        (-hf64!("0x1.3b6786a5a9a69p-17"), -hf64!("0x1.b53dee1a96ca4p-18"), -hf64!("0x1.282e0781f97f5p-124")),
        (-hf64!("0x1.7b388eb924102p-18"), -hf64!("0x1.06dafba21afffp-18"), hf64!("0x1.fffffffffffffp-72")),
        (-hf64!("0x1.98aae9950914ep-19"), -hf64!("0x1.1b443a4805f8fp-19"), hf64!("0x1.fffffffffffffp-73")),
        (-hf64!("0x1.41e4bec9bc547p-19"), -hf64!("0x1.be3d238468f35p-20"), -hf64!("0x1.fffffffffffffp-74")),
        (-hf64!("0x1.0f1c08e43f217p-20"), -hf64!("0x1.77d66368613b4p-21"), hf64!("0x1.0d3de71c269fcp-130")),
        (-hf64!("0x1.0d296993d1368p-20"), -hf64!("0x1.752326c780f68p-21"), hf64!("0x1.1cade8dbbd5d8p-130")),
        (-hf64!("0x1.8288f6bb77d79p-23"), -hf64!("0x1.0becf6ad924bfp-23"), -hf64!("0x1.fffffffffffffp-77")),
        (-hf64!("0x1.0ee8225c19555p-23"), -hf64!("0x1.778e77e8b1e13p-24"), -hf64!("0x1.3da14df43e675p-129")),
        (-hf64!("0x1.922076a30742p-24"), -hf64!("0x1.16bba98a001c1p-24"), hf64!("0x1.d90eddeada3e4p-133")),
        (-hf64!("0x1.3b6203ff2cbe6p-29"), -hf64!("0x1.b536a7dace8cap-30"), hf64!("0x1.07bb47158694fp-137")),
        (-hf64!("0x1.e07a2fbc7fd84p-30"), -hf64!("0x1.4d0a9e634dc35p-30"), -hf64!("0x1.fffffffffffffp-84")),
        (-hf64!("0x1.ba24ff5dea796p-34"), -hf64!("0x1.3278a26ed0162p-34"), hf64!("0x1.f837135341102p-142")),
        (-hf64!("0x1.d091d774b141ep-35"), -hf64!("0x1.4203e2685b069p-35"), -hf64!("0x1.a166cdf05ff79p-143")),
        (-hf64!("0x1.bfc1e9b0f73aep-35"), -hf64!("0x1.365ca0d933491p-35"), -hf64!("0x1.fffffffffffffp-89")),
        (-hf64!("0x1.6d752e77a9dc2p-41"), -hf64!("0x1.faa1cb0d787b3p-42"), -hf64!("0x1.fffffffffffffp-96")),
        (-hf64!("0x1.a1f242e670d73p-42"), -hf64!("0x1.21b2c54479c4dp-42"), hf64!("0x1.48a5d173822e8p-150")),
        (-hf64!("0x1.66ff2474821a7p-44"), -hf64!("0x1.f1accede78f51p-45"), -hf64!("0x1.fffffffffffffp-99")),
        (-hf64!("0x1.3f596b7b1c5e2p-44"), -hf64!("0x1.bab64e105226ap-45"), -hf64!("0x1.1a2829f825272p-154")),
        (-hf64!("0x1.daaead688f65fp-45"), -hf64!("0x1.4906541fb8d5cp-45"), -hf64!("0x1.31de54e6dc4ep-151")),
        (-hf64!("0x1.0e5678a7b8bap-47"), -hf64!("0x1.76c48a7a527bp-48"), -hf64!("0x1.6fbb13d1e3faap-155")),
        (-hf64!("0x1.a6d6a49f2187fp-50"), -hf64!("0x1.2516dafdf17adp-50"), -hf64!("0x1.85a87dc0b88p-157")),
        (-hf64!("0x1.2716da024f6d9p-50"), -hf64!("0x1.9914a112c8dadp-51"), hf64!("0x1.004287d8393fep-159")),
        (-hf64!("0x1.2c506b0368099p-51"), -hf64!("0x1.a052e3d5e791p-52"), -hf64!("0x1.428e13d1da7cap-159")),
        (hf64!("0x1.391609b20beaap-51"), hf64!("0x1.b2078ba8f6835p-52"), -hf64!("0x1.85099d92fa919p-162")),
        (hf64!("0x1.4bbbd21c8c721p-51"), hf64!("0x1.cbe169f09f95ep-52"), -hf64!("0x1.256d63281c5efp-158")),
        (hf64!("0x1.29d3990338b4ep-50"), hf64!("0x1.9ce011cf5f46ep-51"), -hf64!("0x1.ba552cb13d27bp-161")),
        (hf64!("0x1.d57070df38af8p-50"), hf64!("0x1.4563f61023f7p-50"), -hf64!("0x1.ac66c9a5a19p-157")),
        (hf64!("0x1.f6ac7a2928816p-49"), hf64!("0x1.5c6d4854f6b3bp-49"), hf64!("0x1.fffffffffffffp-103")),
        (hf64!("0x1.c2ff91c241999p-47"), hf64!("0x1.389bb3cfc42c2p-47"), -hf64!("0x1.023c49dbbc653p-154")),
        (hf64!("0x1.ac65fdb418f72p-40"), hf64!("0x1.28f171f05f6fap-40"), -hf64!("0x1.51612679f6f55p-146")),
        (hf64!("0x1.f06778562781fp-38"), hf64!("0x1.5814c6c101c7p-38"), -hf64!("0x1.333ad12f679a5p-145")),
        (hf64!("0x1.02cdc0b0f314cp-37"), hf64!("0x1.66c7342a5b89fp-38"), -hf64!("0x1.fffffffffffffp-92")),
        (hf64!("0x1.2162f23d082d5p-36"), hf64!("0x1.912cc5483c4bbp-37"), -hf64!("0x1.9b93a2f7e5da4p-145")),
        (hf64!("0x1.2d664653786ecp-36"), hf64!("0x1.a1d414c603907p-37"), hf64!("0x1.fffffffffffffp-91")),
        (hf64!("0x1.b3665a468351p-35"), hf64!("0x1.2dcbd0c2b922fp-35"), -hf64!("0x1.fffffffffffffp-89")),
        (hf64!("0x1.3dc2574084694p-34"), hf64!("0x1.b881f93d3d7bcp-35"), hf64!("0x1.4d32033399826p-141")),
        (hf64!("0x1.b988be78bfb6bp-33"), hf64!("0x1.320c53ed5fb63p-33"), -hf64!("0x1.fffffffffffffp-87")),
        (hf64!("0x1.e755f4897c124p-27"), hf64!("0x1.51cba0164bed2p-27"), -hf64!("0x1.16a01564a1359p-134")),
        (hf64!("0x1.e4c5c9414182ep-26"), hf64!("0x1.5004cdd3ec269p-26"), -hf64!("0x1.411c828b7c78ep-133")),
        (hf64!("0x1.1c4baf7475e2dp-23"), hf64!("0x1.8a1e1272b4997p-24"), hf64!("0x1.fffffffffffffp-78")),
        (hf64!("0x1.3a6de04d8d384p-22"), hf64!("0x1.b3e437cdb56dep-23"), hf64!("0x1p-76")),
        (hf64!("0x1.4753a08baf7fbp-22"), hf64!("0x1.c5c56aeb3c161p-23"), hf64!("0x1.fffffffffffffp-77")),
        (hf64!("0x1.b6275c496a3d5p-16"), hf64!("0x1.2fb5318232c0cp-16"), hf64!("0x1.666c5e4de0763p-123")),
        (hf64!("0x1.02df58d57f81fp-13"), hf64!("0x1.66e3866f48983p-14"), -hf64!("0x1.3e42ea0bb96a5p-121")),
        (hf64!("0x1.a949818632705p-13"), hf64!("0x1.26ceaaf78e39fp-13"), -hf64!("0x1.fffffffffffffp-67")),
        (hf64!("0x1.dcc98bad34bd4p-13"), hf64!("0x1.4a82829ba33efp-13"), hf64!("0x1.3a222409353fep-120")),
        (hf64!("0x1.8ff59817c7989p-9"), hf64!("0x1.15862abd340e8p-9"), hf64!("0x1.6ab813a41d504p-116")),
        (hf64!("0x1.6a8ee6a3521fdp-6"), hf64!("0x1.fa7ca1e176885p-7"), hf64!("0x1.6530cb0e0ba64p-113")),
        (hf64!("0x1.d4a8ebce833a3p-4"), hf64!("0x1.52145769df4fbp-4"), -hf64!("0x1.5d8026da33cabp-111")),
    ];

    if let Some(result) = lookup_exception(&EXCEPTIONS_TABLE, x) {
        return result;
    }

    let x2 = x * x;
    let x4 = x2 * x2;

    let c13 = Q[20].fma(x, Q[19]); // degree 13
    let c11 = Q[18].fma(x, Q[17]); // degree 11
    let c13 = Q[21].fma(x2, c13); // degree 13

    // add Q[16]*x+c11*x2+c13*x4 to Q[15] (degree 9)
    let (hi, lo) = fast_two_sum(Q[15], Q[16] * x + c11 * x2 + c13 * x4);
    // multiply hi+lo by x and add Q[14] (degree 8)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[14], hi);
    let lo = lo + t;

    // multiply hi+lo by x and add Q[12]+Q[13] (degree 7)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[12], hi);
    let lo = lo + (t + Q[13]);

    // multiply hi+lo by x and add Q[10]+Q[11] (degree 6)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[10], hi);
    let lo = lo + (t + Q[11]);

    // multiply hi+lo by x and add Q[8]+Q[9] (degree 5)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[8], hi);
    let lo = lo + (t + Q[9]);

    // multiply hi+lo by x and add Q[6]+Q[7] (degree 4)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[6], hi);
    let lo = lo + (t + Q[7]);

    // multiply hi+lo by x and add Q[4]+Q[5] (degree 3)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[4], hi);
    let lo = lo + (t + Q[5]);

    // multiply hi+lo by x and add Q[2]+Q[3] (degree 2)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[2], hi);
    let lo = lo + (t + Q[3]);

    // multiply hi+lo by x and add Q[0]+Q[1] (degree 2)
    let (hi, lo) = s_mul(x, hi, lo);
    let (hi, t) = fast_two_sum(Q[0], hi);
    let lo = lo + (t + Q[1]);

    // multiply hi+lo by x
    let (hi, lo) = s_mul(x, hi, lo);

    hi + lo
}

/// Approximation of `exp(x)`, where `x = xh + xl`
///
/// `exp(x)` is approximated by `hi + lo`.
///
/// For the error analysis, we only consider the case where `x^y` does not overflow or underflow.
/// We get:
///
/// `(hi + lo) / exp(xh + xl) = 1 + eps` with `|eps| < 2^-74.139`
///
/// Assumes `|xl/xh| < 2^-23.89` and `|xl| < 2^-14.3486`.
///
/// See analysis before the `exp_1()` call in `cr_pow()`, which proves
/// `|rl| < 2^-23.89 |rh|` (here `xh=rh` and `xl=rl`).
///
/// At output, we also have `0.99985 < hi+lo < 1.99995` and `|lo/hi| < 2^-41.4`.
#[inline]
fn exp_1(xh: f64, xl: f64) -> (f64, f64) {
    // |INVLOG2-2^12/log(2)| < 2^-43.4
    const INVLOG2: f64 = hf64!("0x1.71547652b82fep+12");
    const LOG2H: f64 = hf64!("0x1.62e42fefa39efp-13");
    const LOG2L: f64 = hf64!("0x1.abc9e3b39803fp-68");

    let k = (xh * INVLOG2).roundeven();
    let (kh, kl) = s_mul(k, LOG2H, LOG2L);
    let (yh, yl) = fast_two_sum(xh - kh, xl);
    let yl = yl - kl;

    // Note: k is an integer, this is just a conversion.
    let k = i64::cast_from(k);
    let m = (k >> 12) + 0x3ff;
    let i2 = usize::cast_from((k >> 6) & 0x3f);
    let i1 = usize::cast_from(k & 0x3f);

    let (t1h, t1l) = T1[i2];
    let (t2h, t2l) = T2[i1];
    let (hi, lo) = d_mul(t2h, t2l, t1h, t1l);
    let (qh, ql) = q_1(yh, yl);
    let (mut hi, mut lo) = d_mul(hi, lo, qh, ql);

    // Scale by 2^k. Warning: for x near 1024, we can have k=2^22, thus
    // m = 2047, which encodes Inf
    let mut scale = f64::from_bits(m.unsigned() << 52);
    if m == 0x7ff {
        cold_path();
        hi *= 2.0;
        lo *= 2.0;
        scale = f64::from_bits((m - 1).unsigned() << 52);
    }
    hi *= scale;
    lo *= scale;
    (hi, lo)
}

/// returns a double-double approximation `hi+lo` of `exp(x*log(2))` for `|x| < 745`
#[inline]
fn exp_2(x: f64) -> (f64, f64) {
    let k = (x * hf64!("0x1p12")).roundeven();
    // since |x| <= 745 we have k <= 3051520
    let yh = (-k).fma(hf64!("0x1p-12"), x); // exact, |yh| <= 2^-13
    // now x = k + yh, thus 2^x = 2^k * 2^yh, and we multiply yh by log(2)
    // to use the accurate path of exp()
    let (yh, yl) = s_mul(yh, LN2H, LN2L);

    // Note: k is an integer, this is just a conversion.
    let k = i64::cast_from(k);
    let m = (k >> 12) + 0x3ff;
    let i2 = usize::cast_from((k >> 6) & 0x3f);
    let i1 = usize::cast_from(k & 0x3f);

    let (t1h, t1l) = T1[i2];
    let (t2h, t2l) = T2[i1];
    let (hi, lo) = d_mul(t2h, t2l, t1h, t1l);
    let (qh, ql) = q_2(yh, yl);
    let (mut hi, mut lo) = d_mul(hi, lo, qh, ql);

    // Scale by 2^k. Warning: for x near 1024, we can have k=2^22, thus
    // m = 2047, which encodes Inf
    let mut scale = f64::from_bits(m.unsigned() << 52);
    if m == 0x7ff {
        cold_path();
        hi *= 2.0;
        lo *= 2.0;
        scale = f64::from_bits((m - 1).unsigned() << 52);
    }
    hi *= scale;
    lo *= scale;
    (hi, lo)
}

/// The following is a degree-4 polynomial generated by Sollya for `exp(x)` over
/// `[-0.000130273,0.000130273]` with absolute error `< 2^-74.346`.
const Q_1: [f64; 5] = [
    hf64!("0x1p0"),                // degree 0
    hf64!("0x1p0"),                // degree 1
    hf64!("0x1p-1"),               // degree 2
    hf64!("0x1.5555555995d37p-3"), // degree 3
    hf64!("0x1.55555558489dcp-5"), // degree 4
];

/// Approximation for the fast path of `exp(z)` for `z=zh+zl`, with
/// `|z| < 0.000130273 < 2^-12.88` and `|zl| < 2^-42.6` (assuming `x^y` does not overflow
/// or underflow)
#[inline]
fn q_1(zh: f64, zl: f64) -> (f64, f64) {
    let z = zh + zl;
    let q = Q_1[4].fma(zh, Q_1[3]);
    let q = q.fma(z, Q_1[2]);
    let (hi, lo) = fast_two_sum(Q_1[1], q * z);
    let (hi, lo) = d_mul(zh, zl, hi, lo);
    fast_sum(Q_1[0], hi, lo)
}

// FIXME(expm1): use this directly from `expm1` once ported from CORE-MATH.

/// The following is a degree-7 polynomial generated by Sollya for `exp(z)` over
/// `[-0.000130273,0.000130273]` with absolute error `< 2^-113.218`
/// (see file `exp_accurate.sollya` in the CORE-MATH repo). Since we use this code only for
/// `|x| > 0.125` in `exp2m1(x)`, the corresponding relative error for `exp2m1` is about
/// `2^-113.218/|exp2m1(-0.125)|` which is about `2^-110`.
const Q_2: [f64; 9] = [
    hf64!("0x1p0"),  // degree 0, Q_2[0]
    hf64!("0x1p0"),  // degree 1, Q_2[1]
    hf64!("0x1p-1"), // degree 2, Q_2[2]
    hf64!("0x1.5555555555555p-3"),
    hf64!("0x1.55555555c4d26p-57"), // degree 3, Q_2[3], Q_2[4]
    hf64!("0x1.5555555555555p-5"),  // degree 4, Q_2[5]
    hf64!("0x1.1111111111111p-7"),  // degree 5, Q_2[6]
    hf64!("0x1.6c16c3fbb4213p-10"), // degree 6, Q_2[7]
    hf64!("0x1.a01a023ede0d7p-13"), // degree 7, Q_2[8]
];

/// Approximation for the accurate path of `exp(z)` for `z=zh+zl`, with
/// `|z| < 0.000130273 < 2^-12.88` and `|zl| < 2^-42.6` (assuming `x^y` does not overflow
/// or underflow)
#[inline]
fn q_2(zh: f64, zl: f64) -> (f64, f64) {
    // Let q[0]..q[7] be the coefficients of degree 0..7 of Q_2.
    // The ulp of q[7]*z^7 is at most 2^-155, thus we can compute q[7]*z^7
    // in double precision only.
    // The ulp of q[6]*z^6 is at most 2^-139, thus we can compute q[6]*z^6
    // in double precision only.
    // The ulp of q[5]*z^5 is at most 2^-124, thus we can compute q[5]*z^5
    // in double precision only.
    let z = zh + zl;
    let q = Q_2[8].fma(zh, Q_2[7]);
    let q = q.fma(z, Q_2[6]);
    let q = q.fma(z, Q_2[5]);

    // multiply q by z and add Q_2[3] + Q_2[4]
    let (hi, lo) = a_mul(q, z);
    let (hi, t) = fast_two_sum(Q_2[3], hi);
    let lo = lo + (t + Q_2[4]);

    // multiply hi+lo by zh+zl and add Q_2[2]
    let (hi, lo) = d_mul(hi, lo, zh, zl);
    let (hi, lo) = fast_sum(Q_2[2], hi, lo);
    // multiply hi+lo by zh+zl and add Q_2[1]
    let (hi, lo) = d_mul(hi, lo, zh, zl);
    let (hi, lo) = fast_sum(Q_2[1], hi, lo);
    // multiply hi+lo by zh+zl and add Q_2[0]
    let (hi, lo) = d_mul(hi, lo, zh, zl);
    fast_sum(Q_2[0], hi, lo)
}

/// Add `a + b`, assuming `|a| >= |b|`
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let e = hi - a; // exact
    let lo = b - e; // exact
    (hi, lo)
}

/// Add `a + (bh + bl)`, assuming `|a| >= |bh|`
#[inline]
fn fast_sum(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = fast_two_sum(a, bh);
    // |(a+bh)-(hi+lo)| <= 2^-105 |hi| and |lo| < ulp(hi)
    let lo = lo + bl;
    // |(a+bh+bl)-(hi+lo)| <= 2^-105 |hi| + ulp(lo),
    // where |lo| <= ulp(hi) + |bl|.
    (hi, lo)
}

/// Returns `(ah + al) * (bh + bl) - (al * bl)`
#[inline]
fn d_mul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = a_mul(ah, bh);
    let lo = ah.fma(bl, lo);
    let lo = al.fma(bh, lo);
    (hi, lo)
}

/// Multiply a double with a double double : `a * (bh + bl)`
#[inline]
fn s_mul(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = a_mul(a, bh); // exact
    let lo = a.fma(bl, lo);
    (hi, lo)
}

/// Multiply exactly `a` and `b`, such that `hi + lo = a * b`.
#[inline]
fn a_mul(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    let lo = a.fma(b, -hi);
    (hi, lo)
}

/// Return the rounded sum `hi + lo` for `x` from a table sorted by input value.
/// Each entry contains `(input, hi, lo)`. Return `None` if `x` is absent.
#[inline]
fn lookup_exception(mut table: &[(f64, f64, f64)], x: f64) -> Option<f64> {
    while !table.is_empty() {
        // Invariant: if x is an exceptional case, we have
        // table[0].0 <= x and x <= table[table.len() - 1].0
        let middle = table.len() / 2;
        let (input, hi, lo) = table[middle];

        if x == input {
            return Some(hi + lo);
        }

        if x < input {
            table = &table[..middle];
        } else {
            table = &table[middle + 1..];
        }
    }

    None
}

/// For `0 <= i < 64`, `T1[i] = (h,l)` such that `h+l` is the best double-double approximation
/// of `2^(i/64)`. The approximation error is bounded as follows: `|h + l - 2^(i/64)| < 2^-107`.
#[rustfmt::skip]
static T1: [(f64, f64); 64] = [
    (hf64!("0x1p+0"), hf64!("0x0p+0")),
    (hf64!("0x1.02c9a3e778061p+0"), -hf64!("0x1.19083535b085dp-56")),
    (hf64!("0x1.059b0d3158574p+0"), hf64!("0x1.d73e2a475b465p-55")),
    (hf64!("0x1.0874518759bc8p+0"), hf64!("0x1.186be4bb284ffp-57")),
    (hf64!("0x1.0b5586cf9890fp+0"), hf64!("0x1.8a62e4adc610bp-54")),
    (hf64!("0x1.0e3ec32d3d1a2p+0"), hf64!("0x1.03a1727c57b53p-59")),
    (hf64!("0x1.11301d0125b51p+0"), -hf64!("0x1.6c51039449b3ap-54")),
    (hf64!("0x1.1429aaea92dep+0"), -hf64!("0x1.32fbf9af1369ep-54")),
    (hf64!("0x1.172b83c7d517bp+0"), -hf64!("0x1.19041b9d78a76p-55")),
    (hf64!("0x1.1a35beb6fcb75p+0"), hf64!("0x1.e5b4c7b4968e4p-55")),
    (hf64!("0x1.1d4873168b9aap+0"), hf64!("0x1.e016e00a2643cp-54")),
    (hf64!("0x1.2063b88628cd6p+0"), hf64!("0x1.dc775814a8495p-55")),
    (hf64!("0x1.2387a6e756238p+0"), hf64!("0x1.9b07eb6c70573p-54")),
    (hf64!("0x1.26b4565e27cddp+0"), hf64!("0x1.2bd339940e9d9p-55")),
    (hf64!("0x1.29e9df51fdee1p+0"), hf64!("0x1.612e8afad1255p-55")),
    (hf64!("0x1.2d285a6e4030bp+0"), hf64!("0x1.0024754db41d5p-54")),
    (hf64!("0x1.306fe0a31b715p+0"), hf64!("0x1.6f46ad23182e4p-55")),
    (hf64!("0x1.33c08b26416ffp+0"), hf64!("0x1.32721843659a6p-54")),
    (hf64!("0x1.371a7373aa9cbp+0"), -hf64!("0x1.63aeabf42eae2p-54")),
    (hf64!("0x1.3a7db34e59ff7p+0"), -hf64!("0x1.5e436d661f5e3p-56")),
    (hf64!("0x1.3dea64c123422p+0"), hf64!("0x1.ada0911f09ebcp-55")),
    (hf64!("0x1.4160a21f72e2ap+0"), -hf64!("0x1.ef3691c309278p-58")),
    (hf64!("0x1.44e086061892dp+0"), hf64!("0x1.89b7a04ef80dp-59")),
    (hf64!("0x1.486a2b5c13cdp+0"), hf64!("0x1.3c1a3b69062fp-56")),
    (hf64!("0x1.4bfdad5362a27p+0"), hf64!("0x1.d4397afec42e2p-56")),
    (hf64!("0x1.4f9b2769d2ca7p+0"), -hf64!("0x1.4b309d25957e3p-54")),
    (hf64!("0x1.5342b569d4f82p+0"), -hf64!("0x1.07abe1db13cadp-55")),
    (hf64!("0x1.56f4736b527dap+0"), hf64!("0x1.9bb2c011d93adp-54")),
    (hf64!("0x1.5ab07dd485429p+0"), hf64!("0x1.6324c054647adp-54")),
    (hf64!("0x1.5e76f15ad2148p+0"), hf64!("0x1.ba6f93080e65ep-54")),
    (hf64!("0x1.6247eb03a5585p+0"), -hf64!("0x1.383c17e40b497p-54")),
    (hf64!("0x1.6623882552225p+0"), -hf64!("0x1.bb60987591c34p-54")),
    (hf64!("0x1.6a09e667f3bcdp+0"), -hf64!("0x1.bdd3413b26456p-54")),
    (hf64!("0x1.6dfb23c651a2fp+0"), -hf64!("0x1.bbe3a683c88abp-57")),
    (hf64!("0x1.71f75e8ec5f74p+0"), -hf64!("0x1.16e4786887a99p-55")),
    (hf64!("0x1.75feb564267c9p+0"), -hf64!("0x1.0245957316dd3p-54")),
    (hf64!("0x1.7a11473eb0187p+0"), -hf64!("0x1.41577ee04992fp-55")),
    (hf64!("0x1.7e2f336cf4e62p+0"), hf64!("0x1.05d02ba15797ep-56")),
    (hf64!("0x1.82589994cce13p+0"), -hf64!("0x1.d4c1dd41532d8p-54")),
    (hf64!("0x1.868d99b4492edp+0"), -hf64!("0x1.fc6f89bd4f6bap-54")),
    (hf64!("0x1.8ace5422aa0dbp+0"), hf64!("0x1.6e9f156864b27p-54")),
    (hf64!("0x1.8f1ae99157736p+0"), hf64!("0x1.5cc13a2e3976cp-55")),
    (hf64!("0x1.93737b0cdc5e5p+0"), -hf64!("0x1.75fc781b57ebcp-57")),
    (hf64!("0x1.97d829fde4e5p+0"), -hf64!("0x1.d185b7c1b85d1p-54")),
    (hf64!("0x1.9c49182a3f09p+0"), hf64!("0x1.c7c46b071f2bep-56")),
    (hf64!("0x1.a0c667b5de565p+0"), -hf64!("0x1.359495d1cd533p-54")),
    (hf64!("0x1.a5503b23e255dp+0"), -hf64!("0x1.d2f6edb8d41e1p-54")),
    (hf64!("0x1.a9e6b5579fdbfp+0"), hf64!("0x1.0fac90ef7fd31p-54")),
    (hf64!("0x1.ae89f995ad3adp+0"), hf64!("0x1.7a1cd345dcc81p-54")),
    (hf64!("0x1.b33a2b84f15fbp+0"), -hf64!("0x1.2805e3084d708p-57")),
    (hf64!("0x1.b7f76f2fb5e47p+0"), -hf64!("0x1.5584f7e54ac3bp-56")),
    (hf64!("0x1.bcc1e904bc1d2p+0"), hf64!("0x1.23dd07a2d9e84p-55")),
    (hf64!("0x1.c199bdd85529cp+0"), hf64!("0x1.11065895048ddp-55")),
    (hf64!("0x1.c67f12e57d14bp+0"), hf64!("0x1.2884dff483cadp-54")),
    (hf64!("0x1.cb720dcef9069p+0"), hf64!("0x1.503cbd1e949dbp-56")),
    (hf64!("0x1.d072d4a07897cp+0"), -hf64!("0x1.cbc3743797a9cp-54")),
    (hf64!("0x1.d5818dcfba487p+0"), hf64!("0x1.2ed02d75b3707p-55")),
    (hf64!("0x1.da9e603db3285p+0"), hf64!("0x1.c2300696db532p-54")),
    (hf64!("0x1.dfc97337b9b5fp+0"), -hf64!("0x1.1a5cd4f184b5cp-54")),
    (hf64!("0x1.e502ee78b3ff6p+0"), hf64!("0x1.39e8980a9cc8fp-55")),
    (hf64!("0x1.ea4afa2a490dap+0"), -hf64!("0x1.e9c23179c2893p-54")),
    (hf64!("0x1.efa1bee615a27p+0"), hf64!("0x1.dc7f486a4b6bp-54")),
    (hf64!("0x1.f50765b6e454p+0"), hf64!("0x1.9d3e12dd8a18bp-54")),
    (hf64!("0x1.fa7c1819e90d8p+0"), hf64!("0x1.74853f3a5931ep-55")),
];

/// For `0 <= i < 64`, `T2[i] = (h,l)` such that `h+l` is the best double-double approximation
/// of `2^(i/2^12)`. The approximation error is bounded as follows:
/// `|h + l - 2^(i/2^12)| < 2^-107`.
#[rustfmt::skip]
static T2: [(f64, f64); 64] = [
    (hf64!("0x1p+0"), hf64!("0x0p+0")),
    (hf64!("0x1.000b175effdc7p+0"), hf64!("0x1.ae8e38c59c72ap-54")),
    (hf64!("0x1.00162f3904052p+0"), -hf64!("0x1.7b5d0d58ea8f4p-58")),
    (hf64!("0x1.0021478e11ce6p+0"), hf64!("0x1.4115cb6b16a8ep-54")),
    (hf64!("0x1.002c605e2e8cfp+0"), -hf64!("0x1.d7c96f201bb2fp-55")),
    (hf64!("0x1.003779a95f959p+0"), hf64!("0x1.84711d4c35e9fp-54")),
    (hf64!("0x1.0042936faa3d8p+0"), -hf64!("0x1.0484245243777p-55")),
    (hf64!("0x1.004dadb113dap+0"), -hf64!("0x1.4b237da2025f9p-54")),
    (hf64!("0x1.0058c86da1c0ap+0"), -hf64!("0x1.5e00e62d6b30dp-56")),
    (hf64!("0x1.0063e3a559473p+0"), hf64!("0x1.a1d6cedbb9481p-54")),
    (hf64!("0x1.006eff583fc3dp+0"), -hf64!("0x1.4acf197a00142p-54")),
    (hf64!("0x1.007a1b865a8cap+0"), -hf64!("0x1.eaf2ea42391a5p-57")),
    (hf64!("0x1.0085382faef83p+0"), hf64!("0x1.da93f90835f75p-56")),
    (hf64!("0x1.00905554425d4p+0"), -hf64!("0x1.6a79084ab093cp-55")),
    (hf64!("0x1.009b72f41a12bp+0"), hf64!("0x1.86364f8fbe8f8p-54")),
    (hf64!("0x1.00a6910f3b6fdp+0"), -hf64!("0x1.82e8e14e3110ep-55")),
    (hf64!("0x1.00b1afa5abcbfp+0"), -hf64!("0x1.4f6b2a7609f71p-55")),
    (hf64!("0x1.00bcceb7707ecp+0"), -hf64!("0x1.e1a258ea8f71bp-56")),
    (hf64!("0x1.00c7ee448ee02p+0"), hf64!("0x1.4362ca5bc26f1p-56")),
    (hf64!("0x1.00d30e4d0c483p+0"), hf64!("0x1.095a56c919d02p-54")),
    (hf64!("0x1.00de2ed0ee0f5p+0"), -hf64!("0x1.406ac4e81a645p-57")),
    (hf64!("0x1.00e94fd0398ep+0"), hf64!("0x1.b5a6902767e09p-54")),
    (hf64!("0x1.00f4714af41d3p+0"), -hf64!("0x1.91b2060859321p-54")),
    (hf64!("0x1.00ff93412315cp+0"), hf64!("0x1.427068ab22306p-55")),
    (hf64!("0x1.010ab5b2cbd11p+0"), hf64!("0x1.c1d0660524e08p-54")),
    (hf64!("0x1.0115d89ff3a8bp+0"), -hf64!("0x1.e7bdfb3204be8p-54")),
    (hf64!("0x1.0120fc089ff63p+0"), hf64!("0x1.843aa8b9cbbc6p-55")),
    (hf64!("0x1.012c1fecd613bp+0"), -hf64!("0x1.34104ee7edae9p-56")),
    (hf64!("0x1.0137444c9b5b5p+0"), -hf64!("0x1.2b6aeb6176892p-56")),
    (hf64!("0x1.01426927f5278p+0"), hf64!("0x1.a8cd33b8a1bb3p-56")),
    (hf64!("0x1.014d8e7ee8d2fp+0"), hf64!("0x1.2edc08e5da99ap-56")),
    (hf64!("0x1.0158b4517bb88p+0"), hf64!("0x1.57ba2dc7e0c73p-55")),
    (hf64!("0x1.0163da9fb3335p+0"), hf64!("0x1.b61299ab8cdb7p-54")),
    (hf64!("0x1.016f0169949edp+0"), -hf64!("0x1.90565902c5f44p-54")),
    (hf64!("0x1.017a28af25567p+0"), hf64!("0x1.70fc41c5c2d53p-55")),
    (hf64!("0x1.018550706ab62p+0"), hf64!("0x1.4b9a6e145d76cp-54")),
    (hf64!("0x1.019078ad6a19fp+0"), -hf64!("0x1.008eff5142bf9p-56")),
    (hf64!("0x1.019ba16628de2p+0"), -hf64!("0x1.77669f033c7dep-54")),
    (hf64!("0x1.01a6ca9aac5f3p+0"), -hf64!("0x1.09bb78eeead0ap-54")),
    (hf64!("0x1.01b1f44af9f9ep+0"), hf64!("0x1.371231477ece5p-54")),
    (hf64!("0x1.01bd1e77170b4p+0"), hf64!("0x1.5e7626621eb5bp-56")),
    (hf64!("0x1.01c8491f08f08p+0"), -hf64!("0x1.bc72b100828a5p-54")),
    (hf64!("0x1.01d37442d507p+0"), -hf64!("0x1.ce39cbbab8bbep-57")),
    (hf64!("0x1.01de9fe280ac8p+0"), hf64!("0x1.16996709da2e2p-55")),
    (hf64!("0x1.01e9cbfe113efp+0"), -hf64!("0x1.c11f5239bf535p-55")),
    (hf64!("0x1.01f4f8958c1c6p+0"), hf64!("0x1.e1d4eb5edc6b3p-55")),
    (hf64!("0x1.020025a8f6a35p+0"), -hf64!("0x1.afb99946ee3fp-54")),
    (hf64!("0x1.020b533856324p+0"), -hf64!("0x1.8f06d8a148a32p-54")),
    (hf64!("0x1.02168143b0281p+0"), -hf64!("0x1.2bf310fc54eb6p-55")),
    (hf64!("0x1.0221afcb09e3ep+0"), -hf64!("0x1.c95a035eb4175p-54")),
    (hf64!("0x1.022cdece68c4fp+0"), -hf64!("0x1.491793e46834dp-54")),
    (hf64!("0x1.02380e4dd22adp+0"), -hf64!("0x1.3e8d0d9c49091p-56")),
    (hf64!("0x1.02433e494b755p+0"), -hf64!("0x1.314aa16278aa3p-54")),
    (hf64!("0x1.024e6ec0da046p+0"), hf64!("0x1.48daf888e9651p-55")),
    (hf64!("0x1.02599fb483385p+0"), hf64!("0x1.56dc8046821f4p-55")),
    (hf64!("0x1.0264d1244c719p+0"), hf64!("0x1.45b42356b9d47p-54")),
    (hf64!("0x1.027003103b10ep+0"), -hf64!("0x1.082ef51b61d7ep-56")),
    (hf64!("0x1.027b357854772p+0"), hf64!("0x1.2106ed0920a34p-56")),
    (hf64!("0x1.0286685c9e059p+0"), -hf64!("0x1.fd4cf26ea5d0fp-54")),
    (hf64!("0x1.02919bbd1d1d8p+0"), -hf64!("0x1.09f8775e78084p-54")),
    (hf64!("0x1.029ccf99d720ap+0"), hf64!("0x1.64cbba902ca27p-58")),
    (hf64!("0x1.02a803f2d170dp+0"), hf64!("0x1.4383ef231d207p-54")),
    (hf64!("0x1.02b338c811703p+0"), hf64!("0x1.4a47a505b3a47p-54")),
    (hf64!("0x1.02be6e199c811p+0"), hf64!("0x1.e47120223467fp-54")),
];

#[cfg(test)]
mod tests {
    use super::exp2m1;
    use crate::support::{CastFrom, CastInto, Float};

    #[test]
    fn known_results() {
        for (x, expected) in [
            (f64::ZERO, f64::ZERO),
            (f64::NEG_ZERO, f64::NEG_ZERO),
            (f64::ONE, f64::ONE),
            (f64::NEG_ONE, -0.5),
            (f64::from_bits(1), f64::from_bits(1)),
            (-f64::from_bits(1), -f64::from_bits(1)),
            (f64::from_bits(2), f64::from_bits(1)),
            (-f64::from_bits(2), -f64::from_bits(1)),
            (54.0, hf64!("0x1p+54")),
            (-54.0, f64::NEG_ONE),
            (1023.0, hf64!("0x1p+1023")),
            (1024.0, f64::INFINITY),
            (f64::MAX, f64::INFINITY),
            (f64::MIN, f64::NEG_ONE),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_ONE),
        ] {
            if !cfg!(x86_no_sse2) {
                let result = exp2m1(x);
                assert_biteq!(result, expected, "x = {x:?}");
            }
        }

        for x in [f64::NAN, f64::NEG_NAN, f64::SNAN, f64::NEG_SNAN] {
            // NaN operations can change the sign or payload, including canonicalization on RISC-V.
            assert!(exp2m1(x).is_nan(), "x = {x:?}");
        }
    }

    #[test]
    fn exact_integer_results() {
        for i in 1..=53 {
            let x = f64::cast_from(i);
            let positive = ((1u64 << i) - 1).cast();
            let negative = f64::from_bits((1023 - i) << 52) - 1.0;

            assert_biteq!(exp2m1(x), positive, "x = {x:?}");
            assert_biteq!(exp2m1(-x), negative, "x = {:?}", -x);
        }
    }
}
