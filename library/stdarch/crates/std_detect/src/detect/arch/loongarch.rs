//! Run-time feature detection on LoongArch.

features! {
    @TARGET: loongarch;
    @CFG: target_arch = "loongarch64";
    @MACRO_NAME: is_loongarch_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `loongarch` feature is enabled.
    /// Supported arguments are:
    ///
    /// * `"ual"`
    /// * `"lsx"`
    /// * `"lasx"`
    /// * `"lvz"`
    #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")]
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] ual: "ual";
    /// UAL
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lsx: "lsx";
    /// LSX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lasx: "lasx";
    /// LASX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lvz: "lvz";
    /// LVZ
}
