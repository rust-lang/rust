//! Run-time feature detection on PowerPC.

features! {
    @TARGET: powerpc;
    @MACRO_NAME: is_powerpc_feature_detected;
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
