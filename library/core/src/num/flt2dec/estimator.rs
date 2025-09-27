//! The exponent estimator.

/// Estimates the base-10 scaling factor `k = ceil(log10(mant * 2^exp))`.
///
/// Returns a lower-bound estimate `r` such that:
///
///     k ∈ { r, r + 1 }
#[doc(hidden)]
pub fn estimate_scaling_factor(mant: u64, exp: isize) -> isize {
    // 2^(nbits - 1) < mant <= 2^nbits if mant > 0
    let nbits = 64 - (mant - 1).leading_zeros() as i64;
    let n = nbits + exp as i64;
    //  log₁₀(2ⁿ) = n × log₁₀(2)
    //
    // To multiply with log₁₀(2) as an integer, the fraction (≈0.3) is scaled.
    //
    //  n × log₁₀(2) = (n × log₁₀(2) × C) ÷ C
    //
    // With C = 2³², and ⌊log₁₀(2) × 2³²⌋ = 1292913986, we can compute:
    ((n * 1292913986) >> 32) as isize
}
