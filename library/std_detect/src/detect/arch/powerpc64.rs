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
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power8_altivec: "power8-altivec";
    /// Power8 altivec
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power8_vector: "power8-vector";
    /// Power8 vector
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power8_crypto: "power8-crypto";
    /// Power8 crypto
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power9: "power9";
    without cfg check: true;
    /// Power9
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power9_altivec: "power9-altivec";
    /// Power9 altivec
    @FEATURE: #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")] power9_vector: "power9-vector";
    /// Power9 vector
}
