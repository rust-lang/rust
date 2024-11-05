//! Run-time feature detection on PowerPC64.

features! {
    @TARGET: powerpc64;
    @CFG: target_arch = "powerpc64";
    @MACRO_NAME: is_powerpc64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `powerpc` feature is enabled.
    #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")]
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] altivec: "altivec";
    /// Altivec
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] vsx: "vsx";
    /// VSX
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power8: "power8";
    without cfg check: true;
    /// Power8
}
