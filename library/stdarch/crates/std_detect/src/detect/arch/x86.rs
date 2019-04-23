//! This module implements minimal run-time feature detection for x86.
//!
//! The features are detected using the `detect_features` function below.
//! This function uses the CPUID instruction to read the feature flags from the
//! CPU and encodes them in an `usize` where each bit position represents
//! whether a feature is available (bit is set) or unavaiable (bit is cleared).
//!
//! The enum `Feature` is used to map bit positions to feature names, and the
//! the `__crate::detect::check_for!` macro is used to map string literals (e.g.,
//! "avx") to these bit positions (e.g., `Feature::avx`).
//!
//! The run-time feature detection is performed by the
//! `__crate::detect::check_for(Feature) -> bool` function. On its first call,
//! this functions queries the CPU for the available features and stores them
//! in a global `AtomicUsize` variable. The query is performed by just checking
//! whether the feature bit in this global variable is set or cleared.

features! {
    @TARGET: x86;
    @MACRO_NAME: is_x86_feature_detected;
    @MACRO_ATTRS:
    /// A macro to test at *runtime* whether a CPU feature is available on
    /// x86/x86-64 platforms.
    ///
    /// This macro is provided in the standard library and will detect at runtime
    /// whether the specified CPU feature is detected. This does **not** resolve at
    /// compile time unless the specified feature is already enabled for the entire
    /// crate. Runtime detection currently relies mostly on the `cpuid` instruction.
    ///
    /// This macro only takes one argument which is a string literal of the feature
    /// being tested for. The feature names supported are the lowercase versions of
    /// the ones defined by Intel in [their documentation][docs].
    ///
    /// ## Supported arguments
    ///
    /// This macro supports the same names that `#[target_feature]` supports. Unlike
    /// `#[target_feature]`, however, this macro does not support names separated
    /// with a comma. Instead testing for multiple features must be done through
    /// separate macro invocations for now.
    ///
    /// Supported arguments are:
    ///
    /// * `"aes"`
    /// * `"pclmulqdq"`
    /// * `"rdrand"`
    /// * `"rdseed"`
    /// * `"tsc"`
    /// * `"mmx"`
    /// * `"sse"`
    /// * `"sse2"`
    /// * `"sse3"`
    /// * `"ssse3"`
    /// * `"sse4.1"`
    /// * `"sse4.2"`
    /// * `"sse4a"`
    /// * `"sha"`
    /// * `"avx"`
    /// * `"avx2"`
    /// * `"avx512f"`
    /// * `"avx512cd"`
    /// * `"avx512er"`
    /// * `"avx512pf"`
    /// * `"avx512bw"`
    /// * `"avx512dq"`
    /// * `"avx512vl"`
    /// * `"avx512ifma"`
    /// * `"avx512vbmi"`
    /// * `"avx512vpopcntdq"`
    /// * `"fma"`
    /// * `"bmi1"`
    /// * `"bmi2"`
    /// * `"abm"`
    /// * `"lzcnt"`
    /// * `"tbm"`
    /// * `"popcnt"`
    /// * `"fxsr"`
    /// * `"xsave"`
    /// * `"xsaveopt"`
    /// * `"xsaves"`
    /// * `"xsavec"`
    ///
    /// [docs]: https://software.intel.com/sites/landingpage/IntrinsicsGuide
    #[stable(feature = "simd_x86", since = "1.27.0")]
    @BIND_FEATURE_NAME: "abm"; "lzcnt"; // abm is a synonym for lzcnt
    @FEATURE: aes: "aes";
    /// AES (Advanced Encryption Standard New Instructions AES-NI)
    @FEATURE: pclmulqdq: "pclmulqdq";
    /// CLMUL (Carry-less Multiplication)
    @FEATURE: rdrand: "rdrand";
    /// RDRAND
    @FEATURE: rdseed: "rdseed";
    /// RDSEED
    @FEATURE: tsc: "tsc";
    /// TSC (Time Stamp Counter)
    @FEATURE: mmx: "mmx";
    /// MMX (MultiMedia eXtensions)
    @FEATURE: sse: "sse";
    /// SSE (Streaming SIMD Extensions)
    @FEATURE: sse2: "sse2";
    /// SSE2 (Streaming SIMD Extensions 2)
    @FEATURE: sse3: "sse3";
    /// SSE3 (Streaming SIMD Extensions 3)
    @FEATURE: ssse3: "ssse3";
    /// SSSE3 (Supplemental Streaming SIMD Extensions 3)
    @FEATURE: sse4_1: "sse4.1";
    /// SSE4.1 (Streaming SIMD Extensions 4.1)
    @FEATURE: sse4_2: "sse4.2";
    /// SSE4.2 (Streaming SIMD Extensions 4.2)
    @FEATURE: sse4a: "sse4a";
    /// SSE4a (Streaming SIMD Extensions 4a)
    @FEATURE: sha: "sha";
    /// SHA
    @FEATURE: avx: "avx";
    /// AVX (Advanced Vector Extensions)
    @FEATURE: avx2: "avx2";
    /// AVX2 (Advanced Vector Extensions 2)
    @FEATURE: avx512f: "avx512f" ;
    /// AVX-512 F (Foundation)
    @FEATURE: avx512cd: "avx512cd" ;
    /// AVX-512 CD (Conflict Detection Instructions)
    @FEATURE: avx512er: "avx512er";
    /// AVX-512 ER (Expo nential and Reciprocal Instructions)
    @FEATURE: avx512pf: "avx512pf";
    /// AVX-512 PF (Prefetch Instructions)
    @FEATURE: avx512bw: "avx512bw";
    /// AVX-512 BW (Byte and Word Instructions)
    @FEATURE: avx512dq: "avx512dq";
    /// AVX-512 DQ (Doubleword and Quadword)
    @FEATURE: avx512vl: "avx512vl";
    /// AVX-512 VL (Vector Length Extensions)
    @FEATURE: avx512ifma: "avx512ifma";
    /// AVX-512 IFMA (Integer Fused Multiply Add)
    @FEATURE: avx512vbmi: "avx512vbmi";
    /// AVX-512 VBMI (Vector Byte Manipulation Instructions)
    @FEATURE: avx512vpopcntdq: "avx512vpopcntdq";
    /// AVX-512 VPOPCNTDQ (Vector Population Count Doubleword and
    /// Quadword)
    @FEATURE: f16c: "f16c";
    /// F16C (Conversions between IEEE-754 `binary16` and `binary32` formats)
    @FEATURE: fma: "fma";
    /// FMA (Fused Multiply Add)
    @FEATURE: bmi1: "bmi1" ;
    /// BMI1 (Bit Manipulation Instructions 1)
    @FEATURE: bmi2: "bmi2" ;
    /// BMI2 (Bit Manipulation Instructions 2)
    @FEATURE: lzcnt: "lzcnt";
    /// ABM (Advanced Bit Manipulation) / LZCNT (Leading Zero Count)
    @FEATURE: tbm: "tbm";
    /// TBM (Trailing Bit Manipulation)
    @FEATURE: popcnt: "popcnt";
    /// POPCNT (Population Count)
    @FEATURE: fxsr: "fxsr";
    /// FXSR (Floating-point context fast save and restor)
    @FEATURE: xsave: "xsave";
    /// XSAVE (Save Processor Extended States)
    @FEATURE: xsaveopt: "xsaveopt";
    /// XSAVEOPT (Save Processor Extended States Optimized)
    @FEATURE: xsaves: "xsaves";
    /// XSAVES (Save Processor Extended States Supervisor)
    @FEATURE: xsavec: "xsavec";
    /// XSAVEC (Save Processor Extended States Compacted)
    @FEATURE: cmpxchg16b: "cmpxchg16b";
    /// CMPXCH16B (16-byte compare-and-swap instruction)
    @FEATURE: adx: "adx";
    /// ADX, Intel ADX (Multi-Precision Add-Carry Instruction Extensions)
    @FEATURE: rtm: "rtm";
    /// RTM, Intel (Restricted Transactional Memory)
}
