//! Run-time feature detection on MIPS.

features! {
    @TARGET: mips;
    @MACRO_NAME: is_mips_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `mips` feature is enabled.
    #[unstable(feature = "stdsimd", issue = "27731")]
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
