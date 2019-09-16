//! Run-time feature detection on MIPS64.

features! {
    @TARGET: mips64;
    @MACRO_NAME: is_mips64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `mips64` feature is enabled.
    #[unstable(feature = "stdsimd", issue = "27731")]
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
