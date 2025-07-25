//! This module implements minimal run-time feature detection for x86.
//!
//! The features are detected using the `detect_features` function below.
//! This function uses the CPUID instruction to read the feature flags from the
//! CPU and encodes them in a `usize` where each bit position represents
//! whether a feature is available (bit is set) or unavailable (bit is cleared).
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
    @CFG: any(target_arch = "x86", target_arch = "x86_64");
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
    /// * `"sha512"`
    /// * `"sm3"`
    /// * `"sm4"`
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
    /// * `"avx512vbmi2"`
    /// * `"gfni"`
    /// * `"vaes"`
    /// * `"vpclmulqdq"`
    /// * `"avx512vnni"`
    /// * `"avx512bitalg"`
    /// * `"avx512bf16"`
    /// * `"avx512vp2intersect"`
    /// * `"avx512fp16"`
    /// * `"avxvnni"`
    /// * `"avxifma"`
    /// * `"avxneconvert"`
    /// * `"avxvnniint8"`
    /// * `"avxvnniint16"`
    /// * `"amx-tile"`
    /// * `"amx-int8"`
    /// * `"amx-bf16"`
    /// * `"amx-fp16"`
    /// * `"amx-complex"`
    /// * `"amx-avx512"`
    /// * `"amx-fp8"`
    /// * `"amx-movrs"`
    /// * `"amx-tf32"`
    /// * `"amx-transpose"`
    /// * `"f16c"`
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
    /// * `"cmpxchg16b"`
    /// * `"kl"`
    /// * `"widekl"`
    /// * `"adx"`
    /// * `"rtm"`
    /// * `"movbe"`
    /// * `"ermsb"`
    /// * `"movrs"`
    /// * `"xop"`
    ///
    /// [docs]: https://software.intel.com/sites/landingpage/IntrinsicsGuide
    #[stable(feature = "simd_x86", since = "1.27.0")]
    @BIND_FEATURE_NAME: "abm"; "lzcnt"; // abm is a synonym for lzcnt
    @BIND_FEATURE_NAME: "avx512gfni"; "gfni"; #[deprecated(since = "1.67.0", note = "the `avx512gfni` feature has been renamed to `gfni`")];
    @BIND_FEATURE_NAME: "avx512vaes"; "vaes"; #[deprecated(since = "1.67.0", note = "the `avx512vaes` feature has been renamed to `vaes`")];
    @BIND_FEATURE_NAME: "avx512vpclmulqdq"; "vpclmulqdq"; #[deprecated(since = "1.67.0", note = "the `avx512vpclmulqdq` feature has been renamed to `vpclmulqdq`")];
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] aes: "aes";
    /// AES (Advanced Encryption Standard New Instructions AES-NI)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] pclmulqdq: "pclmulqdq";
    /// CLMUL (Carry-less Multiplication)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] rdrand: "rdrand";
    /// RDRAND
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] rdseed: "rdseed";
    /// RDSEED
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] tsc: "tsc";
    without cfg check: true;
    /// TSC (Time Stamp Counter)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] mmx: "mmx";
    without cfg check: true;
    /// MMX (MultiMedia eXtensions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse: "sse";
    /// SSE (Streaming SIMD Extensions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse2: "sse2";
    /// SSE2 (Streaming SIMD Extensions 2)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse3: "sse3";
    /// SSE3 (Streaming SIMD Extensions 3)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] ssse3: "ssse3";
    /// SSSE3 (Supplemental Streaming SIMD Extensions 3)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse4_1: "sse4.1";
    /// SSE4.1 (Streaming SIMD Extensions 4.1)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse4_2: "sse4.2";
    /// SSE4.2 (Streaming SIMD Extensions 4.2)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sse4a: "sse4a";
    /// SSE4a (Streaming SIMD Extensions 4a)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] sha: "sha";
    /// SHA
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx: "avx";
    /// AVX (Advanced Vector Extensions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx2: "avx2";
    /// AVX2 (Advanced Vector Extensions 2)
    @FEATURE: #[stable(feature = "sha512_sm_x86", since = "1.89.0")] sha512: "sha512";
    /// SHA512
    @FEATURE: #[stable(feature = "sha512_sm_x86", since = "1.89.0")] sm3: "sm3";
    /// SM3
    @FEATURE: #[stable(feature = "sha512_sm_x86", since = "1.89.0")] sm4: "sm4";
    /// SM4
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512f: "avx512f" ;
    /// AVX-512 F (Foundation)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512cd: "avx512cd" ;
    /// AVX-512 CD (Conflict Detection Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512er: "avx512er";
    without cfg check: true;
    /// AVX-512 ER (Expo nential and Reciprocal Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512pf: "avx512pf";
    without cfg check: true;
    /// AVX-512 PF (Prefetch Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512bw: "avx512bw";
    /// AVX-512 BW (Byte and Word Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512dq: "avx512dq";
    /// AVX-512 DQ (Doubleword and Quadword)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vl: "avx512vl";
    /// AVX-512 VL (Vector Length Extensions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512ifma: "avx512ifma";
    /// AVX-512 IFMA (Integer Fused Multiply Add)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vbmi: "avx512vbmi";
    /// AVX-512 VBMI (Vector Byte Manipulation Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vpopcntdq: "avx512vpopcntdq";
    /// AVX-512 VPOPCNTDQ (Vector Population Count Doubleword and Quadword)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vbmi2: "avx512vbmi2";
    /// AVX-512 VBMI2 (Additional byte, word, dword and qword capabilities)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] gfni: "gfni";
    /// AVX-512 GFNI (Galois Field New Instruction)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] vaes: "vaes";
    /// AVX-512 VAES (Vector AES instruction)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] vpclmulqdq: "vpclmulqdq";
    /// AVX-512 VPCLMULQDQ (Vector PCLMULQDQ instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vnni: "avx512vnni";
    /// AVX-512 VNNI (Vector Neural Network Instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512bitalg: "avx512bitalg";
    /// AVX-512 BITALG (Support for VPOPCNT\[B,W\] and VPSHUFBITQMB)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512bf16: "avx512bf16";
    /// AVX-512 BF16 (BFLOAT16 instructions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512vp2intersect: "avx512vp2intersect";
    /// AVX-512 P2INTERSECT
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] avx512fp16: "avx512fp16";
    /// AVX-512 FP16 (FLOAT16 instructions)
    @FEATURE: #[stable(feature = "avx512_target_feature", since = "1.89.0")] avxifma: "avxifma";
    /// AVX-IFMA (Integer Fused Multiply Add)
    @FEATURE: #[stable(feature = "avx512_target_feature", since = "1.89.0")] avxneconvert: "avxneconvert";
    /// AVX-NE-CONVERT (Exceptionless Convert)
    @FEATURE: #[stable(feature = "avx512_target_feature", since = "1.89.0")] avxvnni: "avxvnni";
    /// AVX-VNNI (Vector Neural Network Instructions)
    @FEATURE: #[stable(feature = "avx512_target_feature", since = "1.89.0")] avxvnniint16: "avxvnniint16";
    /// AVX-VNNI_INT8 (VNNI with 16-bit Integers)
    @FEATURE: #[stable(feature = "avx512_target_feature", since = "1.89.0")] avxvnniint8: "avxvnniint8";
    /// AVX-VNNI_INT16 (VNNI with 8-bit integers)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_tile: "amx-tile";
    /// AMX (Advanced Matrix Extensions) - Tile load/store
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_int8: "amx-int8";
    /// AMX-INT8 (Operations on 8-bit integers)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_bf16: "amx-bf16";
    /// AMX-BF16 (BFloat16 Operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_fp16: "amx-fp16";
    /// AMX-FP16 (Float16 Operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_complex: "amx-complex";
    /// AMX-COMPLEX (Complex number Operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_avx512: "amx-avx512";
    /// AMX-AVX512 (AVX512 operations extended to matrices)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_fp8: "amx-fp8";
    /// AMX-FP8 (Float8 Operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_movrs: "amx-movrs";
    /// AMX-MOVRS (Matrix MOVERS operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_tf32: "amx-tf32";
    /// AMX-TF32 (TensorFloat32 Operations)
    @FEATURE: #[unstable(feature = "x86_amx_intrinsics", issue = "126622")] amx_transpose: "amx-transpose";
    /// AMX-TRANSPOSE (Matrix Transpose Operations)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] f16c: "f16c";
    /// F16C (Conversions between IEEE-754 `binary16` and `binary32` formats)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] fma: "fma";
    /// FMA (Fused Multiply Add)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] bmi1: "bmi1" ;
    /// BMI1 (Bit Manipulation Instructions 1)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] bmi2: "bmi2" ;
    /// BMI2 (Bit Manipulation Instructions 2)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] lzcnt: "lzcnt";
    /// ABM (Advanced Bit Manipulation) / LZCNT (Leading Zero Count)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] tbm: "tbm";
    /// TBM (Trailing Bit Manipulation)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] popcnt: "popcnt";
    /// POPCNT (Population Count)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] fxsr: "fxsr";
    /// FXSR (Floating-point context fast save and restore)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] xsave: "xsave";
    /// XSAVE (Save Processor Extended States)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] xsaveopt: "xsaveopt";
    /// XSAVEOPT (Save Processor Extended States Optimized)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] xsaves: "xsaves";
    /// XSAVES (Save Processor Extended States Supervisor)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] xsavec: "xsavec";
    /// XSAVEC (Save Processor Extended States Compacted)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] cmpxchg16b: "cmpxchg16b";
    /// CMPXCH16B (16-byte compare-and-swap instruction)
    @FEATURE: #[stable(feature = "keylocker_x86", since = "1.89.0")] kl: "kl";
    /// Intel Key Locker
    @FEATURE: #[stable(feature = "keylocker_x86", since = "1.89.0")] widekl: "widekl";
    /// Intel Key Locker Wide
    @FEATURE: #[stable(feature = "simd_x86_adx", since = "1.33.0")] adx: "adx";
    /// ADX, Intel ADX (Multi-Precision Add-Carry Instruction Extensions)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] rtm: "rtm";
    /// RTM, Intel (Restricted Transactional Memory)
    @FEATURE: #[stable(feature = "movbe_target_feature", since = "1.67.0")] movbe: "movbe";
    /// MOVBE (Move Data After Swapping Bytes)
    @FEATURE: #[unstable(feature = "movrs_target_feature", issue = "137976")] movrs: "movrs";
    /// MOVRS (Move data with the read-shared hint)
    @FEATURE: #[stable(feature = "simd_x86", since = "1.27.0")] ermsb: "ermsb";
    /// ERMSB, Enhanced REP MOVSB and STOSB
    @FEATURE: #[unstable(feature = "xop_target_feature", issue = "127208")] xop: "xop";
    /// XOP: eXtended Operations (AMD)
}
