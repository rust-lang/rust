//! Run-time feature detection on s390x.

features! {
    @TARGET: s390x;
    @CFG: target_arch = "s390x";
    @MACRO_NAME: is_s390x_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `s390x` feature is enabled.
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] deflate_conversion: "deflate-conversion";
    /// s390x deflate-conversion facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] enhanced_sort: "enhanced-sort";
    /// s390x enhanced-sort facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] guarded_storage: "guarded-storage";
    /// s390x guarded-storage facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] high_word: "high-word";
    /// s390x high-word facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] nnp_assist: "nnp-assist";
    /// s390x nnp-assist facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] transactional_execution: "transactional-execution";
    /// s390x transactional-execution facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector: "vector";
    /// s390x vector facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_enhancements_1: "vector-enhancements-1";
    /// s390x vector-enhancements-1 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_enhancements_2: "vector-enhancements-2";
    /// s390x vector-enhancements-2 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal: "vector-packed-decimal";
    /// s390x vector-packed-decimal facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal_enhancement: "vector-packed-decimal-enhancement";
    /// s390x vector-packed-decimal-enhancement facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal_enhancement_2: "vector-packed-decimal-enhancement-2";
    /// s390x vector-packed-decimal-enhancement-2 facility
}
