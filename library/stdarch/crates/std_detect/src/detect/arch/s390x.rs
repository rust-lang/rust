//! Run-time feature detection on s390x.

features! {
    @TARGET: s390x;
    @CFG: target_arch = "s390x";
    @MACRO_NAME: is_s390x_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `s390x` feature is enabled.
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector: "vector";
    /// s390x vector facility
}
