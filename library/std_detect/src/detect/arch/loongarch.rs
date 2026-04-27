//! Run-time feature detection on LoongArch.

features! {
    @TARGET: loongarch;
    @CFG: any(target_arch = "loongarch32", target_arch = "loongarch64");
    @MACRO_NAME: is_loongarch_feature_detected;
    @MACRO_ATTRS:
    /// Check for the presence of a CPU feature at runtime.
    ///
    /// When the feature is known to be enabled at compile time (e.g. via `-Ctarget-feature`)
    /// the macro expands to `true`.
    ///
    /// Supported arguments are:
    ///
    /// * `"32s"`
    /// * `"f"`
    /// * `"d"`
    /// * `"frecipe"`
    /// * `"div32"`
    /// * `"lsx"`
    /// * `"lasx"`
    /// * `"lam-bh"`
    /// * `"lamcas"`
    /// * `"ld-seq-sa"`
    /// * `"scq"`
    /// * `"lbt"`
    /// * `"lvz"`
    /// * `"ual"`
    #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")]
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] _32s: "32s";
    /// 32S
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] f: "f";
    /// F
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] d: "d";
    /// D
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] frecipe: "frecipe";
    /// Frecipe
    @FEATURE: #[stable(feature = "stdarch_loongarch_div32", since = "CURRENT_RUSTC_VERSION")] div32: "div32";
    /// Div32
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] lsx: "lsx";
    /// LSX
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] lasx: "lasx";
    /// LASX
    @FEATURE: #[stable(feature = "stdarch_loongarch_lam_bh", since = "CURRENT_RUSTC_VERSION")] lam_bh: "lam-bh";
    /// LAM-BH
    @FEATURE: #[stable(feature = "stdarch_loongarch_lamcas", since = "CURRENT_RUSTC_VERSION")] lamcas: "lamcas";
    /// LAM-CAS
    @FEATURE: #[stable(feature = "stdarch_loongarch_ld_seq_sa", since = "CURRENT_RUSTC_VERSION")] ld_seq_sa: "ld-seq-sa";
    /// LD-SEQ-SA
    @FEATURE: #[stable(feature = "stdarch_loongarch_scq", since = "CURRENT_RUSTC_VERSION")] scq: "scq";
    /// SCQ
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] lbt: "lbt";
    /// LBT
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")] lvz: "lvz";
    /// LVZ
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] ual: "ual";
    /// UAL
}
