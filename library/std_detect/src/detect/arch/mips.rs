//! Run-time feature detection on MIPS.

features! {
    @TARGET: mips;
    @CFG: target_arch = "mips";
    @MACRO_NAME: is_mips_feature_detected;
    @MACRO_ATTRS:
    /// Checks whether a `mips` feature is enabled.
    ///
    /// If the feature has been statically enabled for the whole crate (e.g. with
    /// `-Ctarget-feature`), this macro expands to `true`. Otherwise it performs a
    /// runtime check.
    #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
    @FEATURE: #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")] msa: "msa";
    /// MIPS SIMD Architecture (MSA)
}
