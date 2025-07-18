//! Run-time feature detection on MIPS64.

features! {
    @TARGET: mips64;
    @CFG: target_arch = "mips64";
    @MACRO_NAME: is_mips64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `mips64` feature is enabled.
    #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
    @FEATURE: #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
