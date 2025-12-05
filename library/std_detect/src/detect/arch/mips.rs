//! Run-time feature detection on MIPS.

features! {
    @TARGET: mips;
    @CFG: target_arch = "mips";
    @MACRO_NAME: is_mips_feature_detected;
    @MACRO_ATTRS:
    /// Check for the presence of a CPU feature at runtime.
    ///
    /// When the feature is known to be enabled at compile time (e.g. via `-Ctarget-feature`)
    /// the macro expands to `true`.
    #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
    @FEATURE: #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
