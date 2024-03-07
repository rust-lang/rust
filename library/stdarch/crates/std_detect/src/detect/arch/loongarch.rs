//! Run-time feature detection on LoongArch.

features! {
    @TARGET: loongarch;
    @CFG: target_arch = "loongarch64";
    @MACRO_NAME: is_loongarch_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `loongarch` feature is enabled.
    /// Supported arguments are:
    ///
    /// * `"f"`
    /// * `"d"`
    /// * `"frecipe"`
    /// * `"lsx"`
    /// * `"lasx"`
    /// * `"lbt"`
    /// * `"lvz"`
    /// * `"ual"`
    #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")]
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] f: "f";
    /// F
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] d: "d";
    /// D
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] frecipe: "frecipe";
    /// Frecipe
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lsx: "lsx";
    /// LSX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lasx: "lasx";
    /// LASX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lbt: "lbt";
    /// LBT
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lvz: "lvz";
    /// LVZ
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] ual: "ual";
    /// UAL
}
