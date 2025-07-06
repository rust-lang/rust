//! Aarch64 run-time features.

features! {
    @TARGET: aarch64;
    @CFG: any(target_arch = "aarch64", target_arch = "arm64ec");
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
    /// * `"aes"` - FEAT_AES & FEAT_PMULL
    /// * `"asimd"` or "neon" - FEAT_AdvSIMD
    /// * `"bf16"` - FEAT_BF16
    /// * `"bti"` - FEAT_BTI
    /// * `"crc"` - FEAT_CRC
    /// * `"cssc"` - FEAT_CSSC
    /// * `"dit"` - FEAT_DIT
    /// * `"dotprod"` - FEAT_DotProd
    /// * `"dpb"` - FEAT_DPB
    /// * `"dpb2"` - FEAT_DPB2
    /// * `"ecv"` - FEAT_ECV
    /// * `"f32mm"` - FEAT_F32MM
    /// * `"f64mm"` - FEAT_F64MM
    /// * `"faminmax"` - FEAT_FAMINMAX
    /// * `"fcma"` - FEAT_FCMA
    /// * `"fhm"` - FEAT_FHM
    /// * `"flagm"` - FEAT_FLAGM
    /// * `"flagm2"` - FEAT_FLAGM2
    /// * `"fp"` - FEAT_FP
    /// * `"fp16"` - FEAT_FP16
    /// * `"fp8"` - FEAT_FP8
    /// * `"fp8dot2"` - FEAT_FP8DOT2
    /// * `"fp8dot4"` - FEAT_FP8DOT4
    /// * `"fp8fma"` - FEAT_FP8FMA
    /// * `"fpmr"` - FEAT_FPMR
    /// * `"frintts"` - FEAT_FRINTTS
    /// * `"hbc"` - FEAT_HBC
    /// * `"i8mm"` - FEAT_I8MM
    /// * `"jsconv"` - FEAT_JSCVT
    /// * `"lse"` - FEAT_LSE
    /// * `"lse128"` - FEAT_LSE128
    /// * `"lse2"` - FEAT_LSE2
    /// * `"lut"` - FEAT_LUT
    /// * `"mops"` - FEAT_MOPS
    /// * `"mte"` - FEAT_MTE & FEAT_MTE2
    /// * `"paca"` - FEAT_PAuth (address authentication)
    /// * `"pacg"` - FEAT_Pauth (generic authentication)
    /// * `"pauth-lr"` - FEAT_PAuth_LR
    /// * `"pmull"` - FEAT_PMULL
    /// * `"rand"` - FEAT_RNG
    /// * `"rcpc"` - FEAT_LRCPC
    /// * `"rcpc2"` - FEAT_LRCPC2
    /// * `"rcpc3"` - FEAT_LRCPC3
    /// * `"rdm"` - FEAT_RDM
    /// * `"sb"` - FEAT_SB
    /// * `"sha2"` - FEAT_SHA1 & FEAT_SHA256
    /// * `"sha3"` - FEAT_SHA512 & FEAT_SHA3
    /// * `"sm4"` - FEAT_SM3 & FEAT_SM4
    /// * `"sme"` - FEAT_SME
    /// * `"sme-b16b16"` - FEAT_SME_B16B16
    /// * `"sme-f16f16"` - FEAT_SME_F16F16
    /// * `"sme-f64f64"` - FEAT_SME_F64F64
    /// * `"sme-f8f16"` - FEAT_SME_F8F16
    /// * `"sme-f8f32"` - FEAT_SME_F8F32
    /// * `"sme-fa64"` - FEAT_SME_FA64
    /// * `"sme-i16i64"` - FEAT_SME_I16I64
    /// * `"sme-lutv2"` - FEAT_SME_LUTv2
    /// * `"sme2"` - FEAT_SME2
    /// * `"sme2p1"` - FEAT_SME2p1
    /// * `"ssbs"` - FEAT_SSBS & FEAT_SSBS2
    /// * `"ssve-fp8dot2"` - FEAT_SSVE_FP8DOT2
    /// * `"ssve-fp8dot4"` - FEAT_SSVE_FP8DOT4
    /// * `"ssve-fp8fma"` - FEAT_SSVE_FP8FMA
    /// * `"sve"` - FEAT_SVE
    /// * `"sve-b16b16"` - FEAT_SVE_B16B16 (SVE or SME Z-targeting instructions)
    /// * `"sve2"` - FEAT_SVE2
    /// * `"sve2-aes"` - FEAT_SVE_AES & FEAT_SVE_PMULL128 (SVE2 AES crypto)
    /// * `"sve2-bitperm"` - FEAT_SVE2_BitPerm
    /// * `"sve2-sha3"` - FEAT_SVE2_SHA3
    /// * `"sve2-sm4"` - FEAT_SVE2_SM4
    /// * `"sve2p1"` - FEAT_SVE2p1
    /// * `"tme"` - FEAT_TME
    /// * `"wfxt"` - FEAT_WFxT
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
    @NO_RUNTIME_DETECTION: "v8.8a";
    @NO_RUNTIME_DETECTION: "v8.9a";
    @NO_RUNTIME_DETECTION: "v9.1a";
    @NO_RUNTIME_DETECTION: "v9.2a";
    @NO_RUNTIME_DETECTION: "v9.3a";
    @NO_RUNTIME_DETECTION: "v9.4a";
    @NO_RUNTIME_DETECTION: "v9.5a";
    @NO_RUNTIME_DETECTION: "v9a";
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] asimd: "neon";
    /// FEAT_AdvSIMD (Advanced SIMD/NEON)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] pmull: "pmull";
    implied by target_features: ["aes"];
    /// FEAT_PMULL (Polynomial Multiply) - Implied by `aes` target_feature
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fp: "fp";
    implied by target_features: ["neon"];
    /// FEAT_FP (Floating point support) - Implied by `neon` target_feature
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] aes: "aes";
    /// FEAT_AES (AES SIMD instructions) & FEAT_PMULL (PMULL{2}, 64-bit operand variants)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] bf16: "bf16";
    /// FEAT_BF16 (BFloat16 type, plus MM instructions, plus ASIMD support)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] bti: "bti";
    /// FEAT_BTI (Branch Target Identification)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] crc: "crc";
    /// FEAT_CRC32 (Cyclic Redundancy Check)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] cssc: "cssc";
    /// FEAT_CSSC (Common Short Sequence Compression instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dit: "dit";
    /// FEAT_DIT (Data Independent Timing instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dpb: "dpb";
    /// FEAT_DPB (aka dcpop - data cache clean to point of persistence)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dpb2: "dpb2";
    /// FEAT_DPB2 (aka dcpodp - data cache clean to point of deep persistence)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] dotprod: "dotprod";
    /// FEAT_DotProd (Vector Dot-Product - ASIMDDP)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] ecv: "ecv";
    /// FEAT_ECV (Enhanced Counter Virtualization)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] f32mm: "f32mm";
    /// FEAT_F32MM (single-precision matrix multiplication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] f64mm: "f64mm";
    /// FEAT_F64MM (double-precision matrix multiplication)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] faminmax: "faminmax";
    /// FEAT_FAMINMAX (FAMIN and FAMAX SIMD/SVE/SME instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fcma: "fcma";
    /// FEAT_FCMA (float complex number operations)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fhm: "fhm";
    /// FEAT_FHM (fp16 multiplication instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] flagm: "flagm";
    /// FEAT_FLAGM (flag manipulation instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] flagm2: "flagm2";
    /// FEAT_FLAGM2 (flag manipulation instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] fp16: "fp16";
    /// FEAT_FP16 (Half-float support)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] fp8: "fp8";
    /// FEAT_FP8 (F8CVT Instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] fp8dot2: "fp8dot2";
    /// FEAT_FP8DOT2 (F8DP2 Instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] fp8dot4: "fp8dot4";
    /// FEAT_FP8DOT4 (F8DP4 Instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] fp8fma: "fp8fma";
    /// FEAT_FP8FMA (F8FMA Instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] fpmr: "fpmr";
    without cfg check: true;
    /// FEAT_FPMR (Special-purpose AArch64-FPMR register)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] frintts: "frintts";
    /// FEAT_FRINTTS (float to integer rounding instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] hbc: "hbc";
    /// FEAT_HBC (Hinted conditional branches)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] i8mm: "i8mm";
    /// FEAT_I8MM (integer matrix multiplication, plus ASIMD support)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] jsconv: "jsconv";
    /// FEAT_JSCVT (JavaScript float conversion instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] lse: "lse";
    /// FEAT_LSE (Large System Extension - atomics)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] lse128: "lse128";
    /// FEAT_LSE128 (128-bit atomics)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] lse2: "lse2";
    /// FEAT_LSE2 (unaligned and register-pair atomics)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] lut: "lut";
    /// FEAT_LUT (Lookup Table Instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] mops: "mops";
    /// FEAT_MOPS (Standardization of memory operations)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] mte: "mte";
    /// FEAT_MTE & FEAT_MTE2 (Memory Tagging Extension)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] paca: "paca";
    /// FEAT_PAuth (address authentication)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] pacg: "pacg";
    /// FEAT_PAuth (generic authentication)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] pauth_lr: "pauth-lr";
    /// FEAT_PAuth_LR
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rand: "rand";
    /// FEAT_RNG (Random Number Generator)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rcpc: "rcpc";
    /// FEAT_LRCPC (Release consistent Processor consistent)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rcpc2: "rcpc2";
    /// FEAT_LRCPC2 (RCPC with immediate offsets)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] rcpc3: "rcpc3";
    /// FEAT_LRCPC3 (RCPC Instructions v3)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] rdm: "rdm";
    /// FEAT_RDM (Rounding Doubling Multiply - ASIMDRDM)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sb: "sb";
    /// FEAT_SB (speculation barrier)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sha2: "sha2";
    /// FEAT_SHA1 & FEAT_SHA256 (SHA1 & SHA2-256 instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sha3: "sha3";
    /// FEAT_SHA512 & FEAT_SHA3 (SHA2-512 & SHA3 instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sm4: "sm4";
    /// FEAT_SM3 & FEAT_SM4 (SM3 & SM4 instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme: "sme";
    /// FEAT_SME (Scalable Matrix Extension)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme2: "sme2";
    /// FEAT_SME2 (SME Version 2)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme2p1: "sme2p1";
    /// FEAT_SME2p1 (SME Version 2.1)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_b16b16: "sme-b16b16";
    /// FEAT_SME_B16B16
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_f16f16: "sme-f16f16";
    /// FEAT_SME_F16F16 (Non-widening half-precision FP16 to FP16 arithmetic for SME2)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_f64f64: "sme-f64f64";
    /// FEAT_SME_F64F64 (Double-precision floating-point outer product instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_f8f16: "sme-f8f16";
    /// FEAT_SME_F8F16
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_f8f32: "sme-f8f32";
    /// FEAT_SME_F8F32
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_fa64: "sme-fa64";
    /// FEAT_SME_FA64 (Full A64 instruction set support in Streaming SVE mode)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_i16i64: "sme-i16i64";
    /// FEAT_SME_I16I64 (16-bit to 64-bit integer widening outer product instructions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sme_lutv2: "sme-lutv2";
    /// FEAT_SME_LUTv2 (LUTI4 Instruction)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] ssbs: "ssbs";
    /// FEAT_SSBS & FEAT_SSBS2 (speculative store bypass safe)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] ssve_fp8dot2: "ssve-fp8dot2";
    /// FEAT_SSVE_FP8DOT2
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] ssve_fp8dot4: "ssve-fp8dot4";
    /// FEAT_SSVE_FP8DOT4
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] ssve_fp8fma: "ssve-fp8fma";
    /// FEAT_SSVE_FP8FMA
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve: "sve";
    /// FEAT_SVE (Scalable Vector Extension)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2: "sve2";
    /// FEAT_SVE2 (Scalable Vector Extension 2)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sve2p1: "sve2p1";
    /// FEAT_SVE2p1 (Scalable Vector Extension 2.1)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_aes: "sve2-aes";
    /// FEAT_SVE_AES & FEAT_SVE_PMULL128 (SVE2 AES crypto)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] sve_b16b16: "sve-b16b16";
    /// FEAT_SVE_B16B16 (SVE or SME Z-targeting instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_bitperm: "sve2-bitperm";
    /// FEAT_SVE_BitPerm (SVE2 bit permutation instructions)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_sha3: "sve2-sha3";
    /// FEAT_SVE_SHA3 (SVE2 SHA3 crypto)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] sve2_sm4: "sve2-sm4";
    /// FEAT_SVE_SM4 (SVE2 SM4 crypto)
    @FEATURE: #[stable(feature = "simd_aarch64", since = "1.60.0")] tme: "tme";
    /// FEAT_TME (Transactional Memory Extensions)
    @FEATURE: #[unstable(feature = "stdarch_aarch64_feature_detection", issue = "127764")] wfxt: "wfxt";
    /// FEAT_WFxT (WFET and WFIT Instructions)
}
