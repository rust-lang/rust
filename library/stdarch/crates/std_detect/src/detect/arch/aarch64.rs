//! Aarch64 run-time features.

features! {
    @TARGET: aarch64;
    @MACRO_NAME: is_aarch64_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `aarch64` feature is enabled.
    #[unstable(feature = "stdsimd", issue = "27731")]
    @BIND_FEATURE_NAME: "asimd"; "neon";
    @NO_RUNTIME_DETECTION: "ras";
    @NO_RUNTIME_DETECTION: "v8.1a";
    @NO_RUNTIME_DETECTION: "v8.2a";
    @NO_RUNTIME_DETECTION: "v8.3a";
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] asimd: "neon";
    /// ARM Advanced SIMD (ASIMD)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] pmull: "pmull";
    /// Polynomial Multiply
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] fp: "fp";
    /// Floating point support
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] fp16: "fp16";
    /// Half-float support.
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] sve: "sve";
    /// Scalable Vector Extension (SVE)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] crc: "crc";
    /// CRC32 (Cyclic Redundancy Check)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] crypto: "crypto";
    /// Crypto: AES + PMULL + SHA1 + SHA2
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] lse: "lse";
    /// Atomics (Large System Extension)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] rdm: "rdm";
    /// Rounding Double Multiply (ASIMDRDM)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] rcpc: "rcpc";
    /// Release consistent Processor consistent (RcPc)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] dotprod: "dotprod";
    /// Vector Dot-Product (ASIMDDP)
    @FEATURE: #[unstable(feature = "stdsimd", issue = "27731")] tme: "tme";
    /// Transactional Memory Extensions (TME)
}
