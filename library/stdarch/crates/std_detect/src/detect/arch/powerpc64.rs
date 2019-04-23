//! Run-time feature detection on PowerPC64.

features! {
    @TARGET: powerpc64;
    @MACRO_NAME: is_powerpc64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `powerpc` feature is enabled.
    #[unstable(feature = "stdsimd", issue = "27731")]
    @FEATURE: altivec: "altivec";
    /// Altivec
    @FEATURE: vsx: "vsx";
    /// VSX
    @FEATURE: power8: "power8";
    /// Power8
}
