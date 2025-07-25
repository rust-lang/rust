//! Run-time feature detection on s390x.

features! {
    @TARGET: s390x;
    @CFG: target_arch = "s390x";
    @MACRO_NAME: is_s390x_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `s390x` feature is enabled.
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] concurrent_functions: "concurrent-functions";
    /// s390x concurrent-functions facility
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
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension3: "message-security-assist-extension3";
    /// s390x message-security-assist-extension3 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension4: "message-security-assist-extension4";
    /// s390x message-security-assist-extension4 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension5: "message-security-assist-extension5";
    /// s390x message-security-assist-extension5 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension8: "message-security-assist-extension8";
    /// s390x message-security-assist-extension8 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension9: "message-security-assist-extension9";
    /// s390x message-security-assist-extension9 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] message_security_assist_extension12: "message-security-assist-extension12";
    /// s390x message-security-assist-extension12 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] miscellaneous_extensions_2: "miscellaneous-extensions-2";
    /// s390x miscellaneous-extensions-2 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] miscellaneous_extensions_3: "miscellaneous-extensions-3";
    /// s390x miscellaneous-extensions-3 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] miscellaneous_extensions_4: "miscellaneous-extensions-4";
    /// s390x miscellaneous-extensions-4 facility
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
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_enhancements_3: "vector-enhancements-3";
    /// s390x vector-enhancements-3 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal: "vector-packed-decimal";
    /// s390x vector-packed-decimal facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal_enhancement: "vector-packed-decimal-enhancement";
    /// s390x vector-packed-decimal-enhancement facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal_enhancement_2: "vector-packed-decimal-enhancement-2";
    /// s390x vector-packed-decimal-enhancement-2 facility
    #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
    @FEATURE: #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")] vector_packed_decimal_enhancement_3: "vector-packed-decimal-enhancement-3";
    /// s390x vector-packed-decimal-enhancement-3 facility
}
