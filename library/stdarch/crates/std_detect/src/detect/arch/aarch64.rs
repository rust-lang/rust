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
    @FEATURE: asimd: "neon";
    /// ARM Advanced SIMD (ASIMD)
    @FEATURE: pmull: "pmull";
    /// Polynomial Multiply
    @FEATURE: fp: "fp";
    /// Floating point support
    @FEATURE: fp16: "fp16";
    /// Half-float support.
    @FEATURE: sve: "sve";
    /// Scalable Vector Extension (SVE)
    @FEATURE: crc: "crc";
    /// CRC32 (Cyclic Redundancy Check)
    @FEATURE: crypto: "crypto";
    /// Crypto: AES + PMULL + SHA1 + SHA2
    @FEATURE: lse: "lse";
    /// Atomics (Large System Extension)
    @FEATURE: rdm: "rdm";
    /// Rounding Double Multiply (ASIMDRDM)
    @FEATURE: rcpc: "rcpc";
    /// Release consistent Processor consistent (RcPc)
    @FEATURE: dotprod: "dotprod";
    /// Vector Dot-Product (ASIMDDP)
}
