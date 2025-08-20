//! Run-time feature detection on s390x.

features! {
    @TARGET: s390x;
    @CFG: target_arch = "s390x";
    @MACRO_NAME: is_s390x_feature_detected;
    @MACRO_ATTRS:
    /// Check for the presence of a CPU feature at runtime.
    ///
    /// When the feature is known to be enabled at compile time (e.g. via `-Ctarget-feature`)
    /// the macro expands to `true`.
    #[stable(feature = "stdarch_s390x_feature_detection", since = "CURRENT_RUSTC_VERSION")]
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] concurrent_functions: "concurrent-functions";
    /// s390x concurrent-functions facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] deflate_conversion: "deflate-conversion";
    /// s390x deflate-conversion facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] enhanced_sort: "enhanced-sort";
    /// s390x enhanced-sort facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] guarded_storage: "guarded-storage";
    /// s390x guarded-storage facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] high_word: "high-word";
    /// s390x high-word facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension3: "message-security-assist-extension3";
    /// s390x message-security-assist-extension3 facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension4: "message-security-assist-extension4";
    /// s390x message-security-assist-extension4 facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension5: "message-security-assist-extension5";
    /// s390x message-security-assist-extension5 facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension8: "message-security-assist-extension8";
    /// s390x message-security-assist-extension8 facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension9: "message-security-assist-extension9";
    /// s390x message-security-assist-extension9 facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] message_security_assist_extension12: "message-security-assist-extension12";
    /// s390x message-security-assist-extension12 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] miscellaneous_extensions_2: "miscellaneous-extensions-2";
    /// s390x miscellaneous-extensions-2 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] miscellaneous_extensions_3: "miscellaneous-extensions-3";
    /// s390x miscellaneous-extensions-3 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] miscellaneous_extensions_4: "miscellaneous-extensions-4";
    /// s390x miscellaneous-extensions-4 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] nnp_assist: "nnp-assist";
    /// s390x nnp-assist facility
    @FEATURE: #[unstable(feature = "s390x_target_feature", issue = "44839")] transactional_execution: "transactional-execution";
    /// s390x transactional-execution facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector: "vector";
    /// s390x vector facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_enhancements_1: "vector-enhancements-1";
    /// s390x vector-enhancements-1 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_enhancements_2: "vector-enhancements-2";
    /// s390x vector-enhancements-2 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_enhancements_3: "vector-enhancements-3";
    /// s390x vector-enhancements-3 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_packed_decimal: "vector-packed-decimal";
    /// s390x vector-packed-decimal facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_packed_decimal_enhancement: "vector-packed-decimal-enhancement";
    /// s390x vector-packed-decimal-enhancement facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_packed_decimal_enhancement_2: "vector-packed-decimal-enhancement-2";
    /// s390x vector-packed-decimal-enhancement-2 facility
    @FEATURE: #[stable(feature = "s390x_target_feature_vector", since = "CURRENT_RUSTC_VERSION")] vector_packed_decimal_enhancement_3: "vector-packed-decimal-enhancement-3";
    /// s390x vector-packed-decimal-enhancement-3 facility
}
