//! Run-time feature detection on MIPS.

features! {
    @TARGET: mips;
    @CFG: target_arch = "mips";
    @MACRO_NAME: is_mips_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `mips` feature is enabled.
    #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
    @FEATURE: #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
