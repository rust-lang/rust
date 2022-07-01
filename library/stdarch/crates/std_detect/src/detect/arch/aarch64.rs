//! Aarch64 run-time features.

features! {
    @TARGET: aarch64;
    @CFG: target_arch = "aarch64";
    @MACRO_NAME: is_aarch64_feature_detected;
    @MACRO_ATTRS:
    /// This macro tests, at runtime, whether an `aarch64` feature is enabled on aarch64 platforms.
    /// Currently most features are only supported on linux-based platforms.
    ///
    /// This macro takes one argument which is a string literal of the feature being tested for.
    /// The feature names are mostly taken from their FEAT_* definitions in the [ARM Architecture
    /// Reference Manual][docs].
    ///
    /// ## Supported arguments
    ///
    /// * `"asimd"` or "neon" - FEAT_AdvSIMD
    /// * `"pmull"` - FEAT_PMULL
    /// * `"fp"` - FEAT_FP
    /// * `"fp16"` - FEAT_FP16
    /// * `"sve"` - FEAT_SVE
    /// * `"crc"` - FEAT_CRC
    /// * `"lse"` - FEAT_LSE
    /// * `"lse2"` - FEAT_LSE2
    /// * `"rdm"` - FEAT_RDM
    /// * `"rcpc"` - FEAT_LRCPC
    /// * `"rcpc2"` - FEAT_LRCPC2
    /// * `"dotprod"` - FEAT_DotProd
    /// * `"tme"` - FEAT_TME
    /// * `"fhm"` - FEAT_FHM
    /// * `"dit"` - FEAT_DIT
    /// * `"flagm"` - FEAT_FLAGM
    /// * `"ssbs"` - FEAT_SSBS
    /// * `"sb"` - FEAT_SB
    /// * `"paca"` - FEAT_PAuth (address authentication)
    /// * `"pacg"` - FEAT_Pauth (generic authentication)
    /// * `"dpb"` - FEAT_DPB
    /// * `"dpb2"` - FEAT_DPB2
    /// * `"sve2"` - FEAT_SVE2
    /// * `"sve2-aes"` - FEAT_SVE2_AES
    /// * `"sve2-sm4"` - FEAT_SVE2_SM4
    /// * `"sve2-sha3"` - FEAT_SVE2_SHA3
    /// * `"sve2-bitperm"` - FEAT_SVE2_BitPerm
    /// * `"frintts"` - FEAT_FRINTTS
    /// * `"i8mm"` - FEAT_I8MM
    /// * `"f32mm"` - FEAT_F32MM
    /// * `"f64mm"` - FEAT_F64MM
    /// * `"bf16"` - FEAT_BF16
    /// * `"rand"` - FEAT_RNG
    /// * `"bti"` - FEAT_BTI
    /// * `"mte"` - FEAT_MTE
    /// * `"jsconv"` - FEAT_JSCVT
    /// * `"fcma"` - FEAT_FCMA
    /// * `"aes"` - FEAT_AES
    /// * `"sha2"` - FEAT_SHA1 & FEAT_SHA256
    /// * `"sha3"` - FEAT_SHA512 & FEAT_SHA3
    /// * `"sm4"` - FEAT_SM3 & FEAT_SM4
    ///
    /// [docs]: https://developer.arm.com/documentation/ddi0487/latest
    #[stable(feature = "simd_aarch64", since = "1.60.0")]
    @BIND_FEATURE_NAME: "asimd"; "neon";
    @NO_RUNTIME_DETECTION: "ras";
    @NO_RUNTIME_DETECTION: "v8.1a";
    @NO_RUNTIME_DETECTION: "v8.2a";
    @NO_RUNTIME_DETECTION: "v8.3a";
    @NO_RUNTIME_DETECTION: "v8.4a";
    @NO_RUNTIME_DETECTION: "v8.5a";
    @NO_RUNTIME_DETECTION: "v8.6a";
    @NO_RUNTIME_DETECTION: "v8.7a";
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] asimd: "neon";
    /// FEAT_AdvSIMD (Advanced SIMD/NEON)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] pmull: "pmull";
    /// FEAT_PMULL (Polynomial Multiply)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fp: "fp";
    implied by target_features: ["neon"];
    /// FEAT_FP (Floating point support) - Implied by `neon` target_feature
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fp16: "fp16";
    /// FEAT_FP16 (Half-float support)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve: "sve";
    /// FEAT_SVE (Scalable Vector Extension)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] crc: "crc";
    /// FEAT_CRC32 (Cyclic Redundancy Check)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] lse: "lse";
    /// FEAT_LSE (Large System Extension - atomics)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] lse2: "lse2";
    /// FEAT_LSE2 (unaligned and register-pair atomics)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rdm: "rdm";
    /// FEAT_RDM (Rounding Doubling Multiply - ASIMDRDM)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rcpc: "rcpc";
    /// FEAT_LRCPC (Release consistent Processor consistent)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rcpc2: "rcpc2";
    /// FEAT_LRCPC2 (RCPC with immediate offsets)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dotprod: "dotprod";
    /// FEAT_DotProd (Vector Dot-Product - ASIMDDP)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] tme: "tme";
    /// FEAT_TME (Transactional Memory Extensions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fhm: "fhm";
    /// FEAT_FHM (fp16 multiplication instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dit: "dit";
    /// FEAT_DIT (Data Independent Timing instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] flagm: "flagm";
    /// FEAT_FLAGM (flag manipulation instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] ssbs: "ssbs";
    /// FEAT_SSBS (speculative store bypass safe)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sb: "sb";
    /// FEAT_SB (speculation barrier)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] paca: "paca";
    /// FEAT_PAuth (address authentication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] pacg: "pacg";
    /// FEAT_PAuth (generic authentication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dpb: "dpb";
    /// FEAT_DPB (aka dcpop - data cache clean to point of persistence)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dpb2: "dpb2";
    /// FEAT_DPB2 (aka dcpodp - data cache clean to point of deep persistence)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2: "sve2";
    /// FEAT_SVE2 (Scalable Vector Extension 2)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_aes: "sve2-aes";
    /// FEAT_SVE_AES (SVE2 AES crypto)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_sm4: "sve2-sm4";
    /// FEAT_SVE_SM4 (SVE2 SM4 crypto)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_sha3: "sve2-sha3";
    /// FEAT_SVE_SHA3 (SVE2 SHA3 crypto)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_bitperm: "sve2-bitperm";
    /// FEAT_SVE_BitPerm (SVE2 bit permutation instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] frintts: "frintts";
    /// FEAT_FRINTTS (float to integer rounding instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] i8mm: "i8mm";
    /// FEAT_I8MM (integer matrix multiplication, plus ASIMD support)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] f32mm: "f32mm";
    /// FEAT_F32MM (single-precision matrix multiplication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] f64mm: "f64mm";
    /// FEAT_F64MM (double-precision matrix multiplication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] bf16: "bf16";
    /// FEAT_BF16 (BFloat16 type, plus MM instructions, plus ASIMD support)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rand: "rand";
    /// FEAT_RNG (Random Number Generator)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] bti: "bti";
    /// FEAT_BTI (Branch Target Identification)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] mte: "mte";
    /// FEAT_MTE (Memory Tagging Extension)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] jsconv: "jsconv";
    /// FEAT_JSCVT (JavaScript float conversion instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fcma: "fcma";
    /// FEAT_FCMA (float complex number operations)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] aes: "aes";
    /// FEAT_AES (AES instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sha2: "sha2";
    /// FEAT_SHA1 & FEAT_SHA256 (SHA1 & SHA2-256 instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sha3: "sha3";
    /// FEAT_SHA512 & FEAT_SHA3 (SHA2-512 & SHA3 instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sm4: "sm4";
    /// FEAT_SM3 & FEAT_SM4 (SM3 & SM4 instructions)
}
