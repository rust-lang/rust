//! Run-time feature detection on LoongArch.

features! {
    @TARGET: loongarch;
    @CFG: target_arch = "loongarch64";
    @MACRO_NAME: is_loongarch_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `loongarch` feature is enabled.
    /// Supported arguments are:
    ///
    /// * `"lam"`
    /// * `"ual"`
    /// * `"fpu"`
    /// * `"lsx"`
    /// * `"lasx"`
    /// * `"crc32"`
    /// * `"complex"`
    /// * `"crypto"`
    /// * `"lvz"`
    /// * `"lbtx86"`
    /// * `"lbtarm"`
    /// * `"lbtmips"`
    #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")]
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lam: "lam";
    /// LAM
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] ual: "ual";
    /// UAL
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] fpu: "fpu";
    /// FPU
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lsx: "lsx";
    /// LSX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lasx: "lasx";
    /// LASX
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] crc32: "crc32";
    /// FPU
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] complex: "complex";
    /// Complex
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] crypto: "crypto";
    /// Crypto
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lvz: "lvz";
    /// LVZ
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lbtx86: "lbtx86";
    /// LBT.X86
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lbtarm: "lbtarm";
    /// LBT.ARM
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] lbtmips: "lbtmips";
    /// LBT.MIPS
}
