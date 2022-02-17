//! Run-time feature detection on PowerPC64.

features! {
    @TARGET: powerpc64;
    @CFG: target_arch = "powerpc64";
    @MACRO_NAME: is_powerpc64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `powerpc` feature is enabled.
    #[unstable(feature = "stdsimd", issue = "27731")]
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] altivec: "altivec";
    /// Altivec
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] vsx: "vsx";
    /// VSX
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] power8: "power8";
    /// Power8
}
