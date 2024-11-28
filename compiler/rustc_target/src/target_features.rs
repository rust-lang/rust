//! Declares Rust's target feature names for each target.
//! Note that these are similar to but not always identical to LLVM's feature names,
//! and Rust adds some features that do not correspond to LLVM features at all.
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_span::symbol::{Symbol, sym};

/// Features that control behaviour of rustc, rather than the codegen.
/// These exist globally and are not in the target-specific lists below.
pub const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

/// Features that require special handling when passing to LLVM:
/// these are target-specific (i.e., must also be listed in the target-specific list below)
/// but do not correspond to an LLVM target feature.
pub const RUSTC_SPECIAL_FEATURES: &[&str] = &["backchain"];

/// Stability information for target features.
#[derive(Debug, Clone, Copy)]
pub enum Stability {
    /// This target feature is stable, it can be used in `#[target_feature]` and
    /// `#[cfg(target_feature)]`.
    Stable,
    /// This target feature is unstable; using it in `#[target_feature]` or `#[cfg(target_feature)]`
    /// requires enabling the given nightly feature.
    Unstable(Symbol),
    /// This feature can not be set via `-Ctarget-feature` or `#[target_feature]`, it can only be set in the basic
    /// target definition. Used in particular for features that change the floating-point ABI.
    Forbidden { reason: &'static str },
}
use Stability::*;

impl<CTX> HashStable<CTX> for Stability {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Stable => {}
            Unstable(sym) => {
                sym.hash_stable(hcx, hasher);
            }
            Forbidden { .. } => {}
        }
    }
}

impl Stability {
    pub fn is_stable(self) -> bool {
        matches!(self, Stable)
    }

    /// Forbidden features are not supported.
    pub fn is_supported(self) -> bool {
        !matches!(self, Forbidden { .. })
    }
}

// Here we list target features that rustc "understands": they can be used in `#[target_feature]`
// and `#[cfg(target_feature)]`. They also do not trigger any warnings when used with
// `-Ctarget-feature`.
//
// Note that even unstable (and even entirely unlisted) features can be used with `-Ctarget-feature`
// on stable. Using a feature not on the list of Rust target features only emits a warning.
// Only `cfg(target_feature)` and `#[target_feature]` actually do any stability gating.
// `cfg(target_feature)` for unstable features just works on nightly without any feature gate.
// `#[target_feature]` requires a feature gate.
//
// When adding features to the below lists
// check whether they're named already elsewhere in rust
// e.g. in stdarch and whether the given name matches LLVM's
// if it doesn't, to_llvm_feature in llvm_util in rustc_codegen_llvm needs to be adapted.
//
// Also note that all target features listed here must be purely additive: for target_feature 1.1 to
// be sound, we can never allow features like `+soft-float` (on x86) to be controlled on a
// per-function level, since we would then allow safe calls from functions with `+soft-float` to
// functions without that feature!
//
// It is important for soundness that features allowed here do *not* change the function call ABI.
// For example, disabling the `x87` feature on x86 changes how scalar floats are passed as
// arguments, so enabling toggling that feature would be unsound. In fact, since `-Ctarget-feature`
// will just allow unknown features (with a warning), we have to explicitly list features that change
// the ABI as `Forbidden` to ensure using them causes an error. Note that this is only effective if
// such features can never be toggled via `-Ctarget-cpu`! If that is ever a possibility, we will need
// extra checks ensuring that the LLVM-computed target features for a CPU did not (un)set a
// `Forbidden` feature. See https://github.com/rust-lang/rust/issues/116344 for some more context.
// FIXME: add such "forbidden" features for non-x86 targets.
//
// The one exception to features that change the ABI is features that enable larger vector
// registers. Those are permitted to be listed here. This is currently unsound (see
// https://github.com/rust-lang/rust/issues/116558); in the future we will have to ensure that
// functions can only use such vectors as arguments/return types if the corresponding target feature
// is enabled.
//
// Stabilizing a target feature requires t-lang approval.

// If feature A "implies" feature B, then:
// - when A gets enabled (via `-Ctarget-feature` or `#[target_feature]`), we also enable B
// - when B gets disabled (via `-Ctarget-feature`), we also disable A
//
// Both of these are also applied transitively.
type ImpliedFeatures = &'static [&'static str];

const ARM_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("aclass", Unstable(sym::arm_target_feature), &[]),
    ("aes", Unstable(sym::arm_target_feature), &["neon"]),
    ("crc", Unstable(sym::arm_target_feature), &[]),
    ("d32", Unstable(sym::arm_target_feature), &[]),
    ("dotprod", Unstable(sym::arm_target_feature), &["neon"]),
    ("dsp", Unstable(sym::arm_target_feature), &[]),
    ("fp-armv8", Unstable(sym::arm_target_feature), &["vfp4"]),
    ("i8mm", Unstable(sym::arm_target_feature), &["neon"]),
    ("mclass", Unstable(sym::arm_target_feature), &[]),
    ("neon", Unstable(sym::arm_target_feature), &["vfp3"]),
    ("rclass", Unstable(sym::arm_target_feature), &[]),
    ("sha2", Unstable(sym::arm_target_feature), &["neon"]),
    ("soft-float", Forbidden { reason: "unsound because it changes float ABI" }, &[]),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", Unstable(sym::arm_target_feature), &[]),
    ("thumb2", Unstable(sym::arm_target_feature), &[]),
    ("trustzone", Unstable(sym::arm_target_feature), &[]),
    ("v5te", Unstable(sym::arm_target_feature), &[]),
    ("v6", Unstable(sym::arm_target_feature), &["v5te"]),
    ("v6k", Unstable(sym::arm_target_feature), &["v6"]),
    ("v6t2", Unstable(sym::arm_target_feature), &["v6k", "thumb2"]),
    ("v7", Unstable(sym::arm_target_feature), &["v6t2"]),
    ("v8", Unstable(sym::arm_target_feature), &["v7"]),
    ("vfp2", Unstable(sym::arm_target_feature), &[]),
    ("vfp3", Unstable(sym::arm_target_feature), &["vfp2", "d32"]),
    ("vfp4", Unstable(sym::arm_target_feature), &["vfp3"]),
    ("virtualization", Unstable(sym::arm_target_feature), &[]),
    // tidy-alphabetical-end
    // FIXME: need to also forbid turning off `fpregs` on hardfloat targets
];

const AARCH64_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    // FEAT_AES & FEAT_PMULL
    ("aes", Stable, &["neon"]),
    // FEAT_BF16
    ("bf16", Stable, &[]),
    // FEAT_BTI
    ("bti", Stable, &[]),
    // FEAT_CRC
    ("crc", Stable, &[]),
    // FEAT_CSSC
    ("cssc", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_DIT
    ("dit", Stable, &[]),
    // FEAT_DotProd
    ("dotprod", Stable, &["neon"]),
    // FEAT_DPB
    ("dpb", Stable, &[]),
    // FEAT_DPB2
    ("dpb2", Stable, &["dpb"]),
    // FEAT_ECV
    ("ecv", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_F32MM
    ("f32mm", Stable, &["sve"]),
    // FEAT_F64MM
    ("f64mm", Stable, &["sve"]),
    // FEAT_FAMINMAX
    ("faminmax", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_FCMA
    ("fcma", Stable, &["neon"]),
    // FEAT_FHM
    ("fhm", Stable, &["fp16"]),
    // FEAT_FLAGM
    ("flagm", Stable, &[]),
    // FEAT_FLAGM2
    ("flagm2", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_FP16
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("fp16", Stable, &["neon"]),
    // FEAT_FP8
    ("fp8", Unstable(sym::aarch64_unstable_target_feature), &["faminmax", "lut", "bf16"]),
    // FEAT_FP8DOT2
    ("fp8dot2", Unstable(sym::aarch64_unstable_target_feature), &["fp8dot4"]),
    // FEAT_FP8DOT4
    ("fp8dot4", Unstable(sym::aarch64_unstable_target_feature), &["fp8fma"]),
    // FEAT_FP8FMA
    ("fp8fma", Unstable(sym::aarch64_unstable_target_feature), &["fp8"]),
    // FEAT_FRINTTS
    ("frintts", Stable, &[]),
    // FEAT_HBC
    ("hbc", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_I8MM
    ("i8mm", Stable, &[]),
    // FEAT_JSCVT
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("jsconv", Stable, &["neon"]),
    // FEAT_LOR
    ("lor", Stable, &[]),
    // FEAT_LSE
    ("lse", Stable, &[]),
    // FEAT_LSE128
    ("lse128", Unstable(sym::aarch64_unstable_target_feature), &["lse"]),
    // FEAT_LSE2
    ("lse2", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_LUT
    ("lut", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_MOPS
    ("mops", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_MTE & FEAT_MTE2
    ("mte", Stable, &[]),
    // FEAT_AdvSimd & FEAT_FP
    ("neon", Stable, &[]),
    // FEAT_PAUTH (address authentication)
    ("paca", Stable, &[]),
    // FEAT_PAUTH (generic authentication)
    ("pacg", Stable, &[]),
    // FEAT_PAN
    ("pan", Stable, &[]),
    // FEAT_PAuth_LR
    ("pauth-lr", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_PMUv3
    ("pmuv3", Stable, &[]),
    // FEAT_RNG
    ("rand", Stable, &[]),
    // FEAT_RAS & FEAT_RASv1p1
    ("ras", Stable, &[]),
    // FEAT_LRCPC
    ("rcpc", Stable, &[]),
    // FEAT_LRCPC2
    ("rcpc2", Stable, &["rcpc"]),
    // FEAT_LRCPC3
    ("rcpc3", Unstable(sym::aarch64_unstable_target_feature), &["rcpc2"]),
    // FEAT_RDM
    ("rdm", Stable, &["neon"]),
    // FEAT_SB
    ("sb", Stable, &[]),
    // FEAT_SHA1 & FEAT_SHA256
    ("sha2", Stable, &["neon"]),
    // FEAT_SHA512 & FEAT_SHA3
    ("sha3", Stable, &["sha2"]),
    // FEAT_SM3 & FEAT_SM4
    ("sm4", Stable, &["neon"]),
    // FEAT_SME
    ("sme", Unstable(sym::aarch64_unstable_target_feature), &["bf16"]),
    // FEAT_SME_B16B16
    ("sme-b16b16", Unstable(sym::aarch64_unstable_target_feature), &["bf16", "sme2", "sve-b16b16"]),
    // FEAT_SME_F16F16
    ("sme-f16f16", Unstable(sym::aarch64_unstable_target_feature), &["sme2"]),
    // FEAT_SME_F64F64
    ("sme-f64f64", Unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME_F8F16
    ("sme-f8f16", Unstable(sym::aarch64_unstable_target_feature), &["sme-f8f32"]),
    // FEAT_SME_F8F32
    ("sme-f8f32", Unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"]),
    // FEAT_SME_FA64
    ("sme-fa64", Unstable(sym::aarch64_unstable_target_feature), &["sme", "sve2"]),
    // FEAT_SME_I16I64
    ("sme-i16i64", Unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME_LUTv2
    ("sme-lutv2", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_SME2
    ("sme2", Unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME2p1
    ("sme2p1", Unstable(sym::aarch64_unstable_target_feature), &["sme2"]),
    // FEAT_SPE
    ("spe", Stable, &[]),
    // FEAT_SSBS & FEAT_SSBS2
    ("ssbs", Stable, &[]),
    // FEAT_SSVE_FP8FDOT2
    ("ssve-fp8dot2", Unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8dot4"]),
    // FEAT_SSVE_FP8FDOT4
    ("ssve-fp8dot4", Unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8fma"]),
    // FEAT_SSVE_FP8FMA
    ("ssve-fp8fma", Unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"]),
    // FEAT_SVE
    // It was decided that SVE requires Neon: https://github.com/rust-lang/rust/pull/91608
    //
    // LLVM doesn't enable Neon for SVE. ARM indicates that they're separate, but probably always
    // exist together: https://developer.arm.com/documentation/102340/0100/New-features-in-SVE2
    //
    // "For backwards compatibility, Neon and VFP are required in the latest architectures."
    ("sve", Stable, &["neon"]),
    // FEAT_SVE_B16B16 (SVE or SME Z-targeting instructions)
    ("sve-b16b16", Unstable(sym::aarch64_unstable_target_feature), &["bf16"]),
    // FEAT_SVE2
    ("sve2", Stable, &["sve"]),
    // FEAT_SVE_AES & FEAT_SVE_PMULL128
    ("sve2-aes", Stable, &["sve2", "aes"]),
    // FEAT_SVE2_BitPerm
    ("sve2-bitperm", Stable, &["sve2"]),
    // FEAT_SVE2_SHA3
    ("sve2-sha3", Stable, &["sve2", "sha3"]),
    // FEAT_SVE2_SM4
    ("sve2-sm4", Stable, &["sve2", "sm4"]),
    // FEAT_SVE2p1
    ("sve2p1", Unstable(sym::aarch64_unstable_target_feature), &["sve2"]),
    // FEAT_TME
    ("tme", Stable, &[]),
    ("v8.1a", Unstable(sym::aarch64_ver_target_feature), &[
        "crc", "lse", "rdm", "pan", "lor", "vh",
    ]),
    ("v8.2a", Unstable(sym::aarch64_ver_target_feature), &["v8.1a", "ras", "dpb"]),
    ("v8.3a", Unstable(sym::aarch64_ver_target_feature), &[
        "v8.2a", "rcpc", "paca", "pacg", "jsconv",
    ]),
    ("v8.4a", Unstable(sym::aarch64_ver_target_feature), &["v8.3a", "dotprod", "dit", "flagm"]),
    ("v8.5a", Unstable(sym::aarch64_ver_target_feature), &["v8.4a", "ssbs", "sb", "dpb2", "bti"]),
    ("v8.6a", Unstable(sym::aarch64_ver_target_feature), &["v8.5a", "bf16", "i8mm"]),
    ("v8.7a", Unstable(sym::aarch64_ver_target_feature), &["v8.6a", "wfxt"]),
    ("v8.8a", Unstable(sym::aarch64_ver_target_feature), &["v8.7a", "hbc", "mops"]),
    ("v8.9a", Unstable(sym::aarch64_ver_target_feature), &["v8.8a", "cssc"]),
    ("v9.1a", Unstable(sym::aarch64_ver_target_feature), &["v9a", "v8.6a"]),
    ("v9.2a", Unstable(sym::aarch64_ver_target_feature), &["v9.1a", "v8.7a"]),
    ("v9.3a", Unstable(sym::aarch64_ver_target_feature), &["v9.2a", "v8.8a"]),
    ("v9.4a", Unstable(sym::aarch64_ver_target_feature), &["v9.3a", "v8.9a"]),
    ("v9.5a", Unstable(sym::aarch64_ver_target_feature), &["v9.4a"]),
    ("v9a", Unstable(sym::aarch64_ver_target_feature), &["v8.5a", "sve2"]),
    // FEAT_VHE
    ("vh", Stable, &[]),
    // FEAT_WFxT
    ("wfxt", Unstable(sym::aarch64_unstable_target_feature), &[]),
    // tidy-alphabetical-end
];

const AARCH64_TIED_FEATURES: &[&[&str]] = &[
    &["paca", "pacg"], // Together these represent `pauth` in LLVM
];

const X86_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("adx", Stable, &[]),
    ("aes", Stable, &["sse2"]),
    ("amx-bf16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-complex", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-fp16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-int8", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-tile", Unstable(sym::x86_amx_intrinsics), &[]),
    ("avx", Stable, &["sse4.2"]),
    ("avx2", Stable, &["avx"]),
    ("avx512bf16", Unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512bitalg", Unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512bw", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512cd", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512dq", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512f", Unstable(sym::avx512_target_feature), &["avx2", "fma", "f16c"]),
    ("avx512fp16", Unstable(sym::avx512_target_feature), &["avx512bw", "avx512vl", "avx512dq"]),
    ("avx512ifma", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vbmi", Unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512vbmi2", Unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512vl", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vnni", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vp2intersect", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vpopcntdq", Unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avxifma", Unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxneconvert", Unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnni", Unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnniint16", Unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnniint8", Unstable(sym::avx512_target_feature), &["avx2"]),
    ("bmi1", Stable, &[]),
    ("bmi2", Stable, &[]),
    ("cmpxchg16b", Stable, &[]),
    ("ermsb", Unstable(sym::ermsb_target_feature), &[]),
    ("f16c", Stable, &["avx"]),
    ("fma", Stable, &["avx"]),
    ("fxsr", Stable, &[]),
    ("gfni", Unstable(sym::avx512_target_feature), &["sse2"]),
    ("lahfsahf", Unstable(sym::lahfsahf_target_feature), &[]),
    ("lzcnt", Stable, &[]),
    ("movbe", Stable, &[]),
    ("pclmulqdq", Stable, &["sse2"]),
    ("popcnt", Stable, &[]),
    ("prfchw", Unstable(sym::prfchw_target_feature), &[]),
    ("rdrand", Stable, &[]),
    ("rdseed", Stable, &[]),
    ("rtm", Unstable(sym::rtm_target_feature), &[]),
    ("sha", Stable, &["sse2"]),
    ("sha512", Unstable(sym::sha512_sm_x86), &["avx2"]),
    ("sm3", Unstable(sym::sha512_sm_x86), &["avx"]),
    ("sm4", Unstable(sym::sha512_sm_x86), &["avx2"]),
    ("soft-float", Forbidden { reason: "unsound because it changes float ABI" }, &[]),
    ("sse", Stable, &[]),
    ("sse2", Stable, &["sse"]),
    ("sse3", Stable, &["sse2"]),
    ("sse4.1", Stable, &["ssse3"]),
    ("sse4.2", Stable, &["sse4.1"]),
    ("sse4a", Unstable(sym::sse4a_target_feature), &["sse3"]),
    ("ssse3", Stable, &["sse3"]),
    ("tbm", Unstable(sym::tbm_target_feature), &[]),
    ("vaes", Unstable(sym::avx512_target_feature), &["avx2", "aes"]),
    ("vpclmulqdq", Unstable(sym::avx512_target_feature), &["avx", "pclmulqdq"]),
    ("xop", Unstable(sym::xop_target_feature), &[/*"fma4", */ "avx", "sse4a"]),
    ("xsave", Stable, &[]),
    ("xsavec", Stable, &["xsave"]),
    ("xsaveopt", Stable, &["xsave"]),
    ("xsaves", Stable, &["xsave"]),
    // tidy-alphabetical-end
    // FIXME: need to also forbid turning off `x87` on hardfloat targets
];

const HEXAGON_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("hvx", Unstable(sym::hexagon_target_feature), &[]),
    ("hvx-length128b", Unstable(sym::hexagon_target_feature), &["hvx"]),
    // tidy-alphabetical-end
];

const POWERPC_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("altivec", Unstable(sym::powerpc_target_feature), &[]),
    ("partword-atomics", Unstable(sym::powerpc_target_feature), &[]),
    ("power10-vector", Unstable(sym::powerpc_target_feature), &["power9-vector"]),
    ("power8-altivec", Unstable(sym::powerpc_target_feature), &["altivec"]),
    ("power8-vector", Unstable(sym::powerpc_target_feature), &["vsx", "power8-altivec"]),
    ("power9-altivec", Unstable(sym::powerpc_target_feature), &["power8-altivec"]),
    ("power9-vector", Unstable(sym::powerpc_target_feature), &["power8-vector", "power9-altivec"]),
    ("quadword-atomics", Unstable(sym::powerpc_target_feature), &[]),
    ("vsx", Unstable(sym::powerpc_target_feature), &["altivec"]),
    // tidy-alphabetical-end
];

const MIPS_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("fp64", Unstable(sym::mips_target_feature), &[]),
    ("msa", Unstable(sym::mips_target_feature), &[]),
    ("virt", Unstable(sym::mips_target_feature), &[]),
    // tidy-alphabetical-end
];

const RISCV_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("a", Stable, &["zaamo", "zalrsc"]),
    ("c", Stable, &[]),
    ("d", Unstable(sym::riscv_target_feature), &["f"]),
    ("e", Unstable(sym::riscv_target_feature), &[]),
    ("f", Unstable(sym::riscv_target_feature), &[]),
    ("m", Stable, &[]),
    ("relax", Unstable(sym::riscv_target_feature), &[]),
    ("unaligned-scalar-mem", Unstable(sym::riscv_target_feature), &[]),
    ("v", Unstable(sym::riscv_target_feature), &[]),
    ("zaamo", Unstable(sym::riscv_target_feature), &[]),
    ("zabha", Unstable(sym::riscv_target_feature), &["zaamo"]),
    ("zalrsc", Unstable(sym::riscv_target_feature), &[]),
    ("zba", Stable, &[]),
    ("zbb", Stable, &[]),
    ("zbc", Stable, &[]),
    ("zbkb", Stable, &[]),
    ("zbkc", Stable, &[]),
    ("zbkx", Stable, &[]),
    ("zbs", Stable, &[]),
    ("zdinx", Unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zfh", Unstable(sym::riscv_target_feature), &["zfhmin"]),
    ("zfhmin", Unstable(sym::riscv_target_feature), &["f"]),
    ("zfinx", Unstable(sym::riscv_target_feature), &[]),
    ("zhinx", Unstable(sym::riscv_target_feature), &["zhinxmin"]),
    ("zhinxmin", Unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zk", Stable, &["zkn", "zkr", "zkt"]),
    ("zkn", Stable, &["zbkb", "zbkc", "zbkx", "zkne", "zknd", "zknh"]),
    ("zknd", Stable, &[]),
    ("zkne", Stable, &[]),
    ("zknh", Stable, &[]),
    ("zkr", Stable, &[]),
    ("zks", Stable, &["zbkb", "zbkc", "zbkx", "zksed", "zksh"]),
    ("zksed", Stable, &[]),
    ("zksh", Stable, &[]),
    ("zkt", Stable, &[]),
    // tidy-alphabetical-end
];

const WASM_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("atomics", Unstable(sym::wasm_target_feature), &[]),
    ("bulk-memory", Stable, &[]),
    ("exception-handling", Unstable(sym::wasm_target_feature), &[]),
    ("extended-const", Stable, &[]),
    ("multivalue", Stable, &[]),
    ("mutable-globals", Stable, &[]),
    ("nontrapping-fptoint", Stable, &[]),
    ("reference-types", Stable, &[]),
    ("relaxed-simd", Stable, &["simd128"]),
    ("sign-ext", Stable, &[]),
    ("simd128", Stable, &[]),
    ("tail-call", Stable, &[]),
    ("wide-arithmetic", Unstable(sym::wasm_target_feature), &[]),
    // tidy-alphabetical-end
];

const BPF_FEATURES: &[(&str, Stability, ImpliedFeatures)] =
    &[("alu32", Unstable(sym::bpf_target_feature), &[])];

const CSKY_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("10e60", Unstable(sym::csky_target_feature), &["7e10"]),
    ("2e3", Unstable(sym::csky_target_feature), &["e2"]),
    ("3e3r1", Unstable(sym::csky_target_feature), &[]),
    ("3e3r2", Unstable(sym::csky_target_feature), &["3e3r1", "doloop"]),
    ("3e3r3", Unstable(sym::csky_target_feature), &["doloop"]),
    ("3e7", Unstable(sym::csky_target_feature), &["2e3"]),
    ("7e10", Unstable(sym::csky_target_feature), &["3e7"]),
    ("cache", Unstable(sym::csky_target_feature), &[]),
    ("doloop", Unstable(sym::csky_target_feature), &[]),
    ("dsp1e2", Unstable(sym::csky_target_feature), &[]),
    ("dspe60", Unstable(sym::csky_target_feature), &[]),
    ("e1", Unstable(sym::csky_target_feature), &["elrw"]),
    ("e2", Unstable(sym::csky_target_feature), &["e2"]),
    ("edsp", Unstable(sym::csky_target_feature), &[]),
    ("elrw", Unstable(sym::csky_target_feature), &[]),
    ("float1e2", Unstable(sym::csky_target_feature), &[]),
    ("float1e3", Unstable(sym::csky_target_feature), &[]),
    ("float3e4", Unstable(sym::csky_target_feature), &[]),
    ("float7e60", Unstable(sym::csky_target_feature), &[]),
    ("floate1", Unstable(sym::csky_target_feature), &[]),
    ("hard-tp", Unstable(sym::csky_target_feature), &[]),
    ("high-registers", Unstable(sym::csky_target_feature), &[]),
    ("hwdiv", Unstable(sym::csky_target_feature), &[]),
    ("mp", Unstable(sym::csky_target_feature), &["2e3"]),
    ("mp1e2", Unstable(sym::csky_target_feature), &["3e7"]),
    ("nvic", Unstable(sym::csky_target_feature), &[]),
    ("trust", Unstable(sym::csky_target_feature), &[]),
    ("vdsp2e60f", Unstable(sym::csky_target_feature), &[]),
    ("vdspv1", Unstable(sym::csky_target_feature), &[]),
    ("vdspv2", Unstable(sym::csky_target_feature), &[]),
    // tidy-alphabetical-end
    //fpu
    // tidy-alphabetical-start
    ("fdivdu", Unstable(sym::csky_target_feature), &[]),
    ("fpuv2_df", Unstable(sym::csky_target_feature), &[]),
    ("fpuv2_sf", Unstable(sym::csky_target_feature), &[]),
    ("fpuv3_df", Unstable(sym::csky_target_feature), &[]),
    ("fpuv3_hf", Unstable(sym::csky_target_feature), &[]),
    ("fpuv3_hi", Unstable(sym::csky_target_feature), &[]),
    ("fpuv3_sf", Unstable(sym::csky_target_feature), &[]),
    ("hard-float", Unstable(sym::csky_target_feature), &[]),
    ("hard-float-abi", Unstable(sym::csky_target_feature), &[]),
    // tidy-alphabetical-end
];

const LOONGARCH_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("d", Unstable(sym::loongarch_target_feature), &["f"]),
    ("f", Unstable(sym::loongarch_target_feature), &[]),
    ("frecipe", Unstable(sym::loongarch_target_feature), &[]),
    ("lasx", Unstable(sym::loongarch_target_feature), &["lsx"]),
    ("lbt", Unstable(sym::loongarch_target_feature), &[]),
    ("lsx", Unstable(sym::loongarch_target_feature), &["d"]),
    ("lvz", Unstable(sym::loongarch_target_feature), &[]),
    ("relax", Unstable(sym::loongarch_target_feature), &[]),
    ("ual", Unstable(sym::loongarch_target_feature), &[]),
    // tidy-alphabetical-end
];

const IBMZ_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("backchain", Unstable(sym::s390x_target_feature), &[]),
    ("vector", Unstable(sym::s390x_target_feature), &[]),
    // tidy-alphabetical-end
];

const SPARC_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("leoncasa", Unstable(sym::sparc_target_feature), &[]),
    ("v8plus", Unstable(sym::sparc_target_feature), &[]),
    ("v9", Unstable(sym::sparc_target_feature), &[]),
    // tidy-alphabetical-end
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primitives may be documented.
///
/// IMPORTANT: If you're adding another feature list above, make sure to add it to this iterator!
pub fn all_rust_features() -> impl Iterator<Item = (&'static str, Stability)> {
    std::iter::empty()
        .chain(ARM_FEATURES.iter())
        .chain(AARCH64_FEATURES.iter())
        .chain(X86_FEATURES.iter())
        .chain(HEXAGON_FEATURES.iter())
        .chain(POWERPC_FEATURES.iter())
        .chain(MIPS_FEATURES.iter())
        .chain(RISCV_FEATURES.iter())
        .chain(WASM_FEATURES.iter())
        .chain(BPF_FEATURES.iter())
        .chain(CSKY_FEATURES)
        .chain(LOONGARCH_FEATURES)
        .chain(IBMZ_FEATURES)
        .chain(SPARC_FEATURES)
        .cloned()
        .map(|(f, s, _)| (f, s))
}

// These arrays represent the least-constraining feature that is required for vector types up to a
// certain size to have their "proper" ABI on each architecture.
// Note that they must be kept sorted by vector size.
const X86_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[(128, "sse"), (256, "avx"), (512, "avx512f")]; // FIXME: might need changes for AVX10.
const AARCH64_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "neon")];

// We might want to add "helium" too.
const ARM_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "neon")];

const POWERPC_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "altivec")];
const WASM_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "simd128")];
const S390X_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "vector")];
const RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[/*(64, "zvl64b"), */ (128, "v")];
// Always warn on SPARC, as the necessary target features cannot be enabled in Rust at the moment.
const SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[/*(64, "vis")*/];

const HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[/*(512, "hvx-length64b"),*/ (1024, "hvx-length128b")];
const MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "msa")];
const CSKY_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "vdspv1")];

impl super::spec::Target {
    pub fn rust_target_features(&self) -> &'static [(&'static str, Stability, ImpliedFeatures)] {
        match &*self.arch {
            "arm" => ARM_FEATURES,
            "aarch64" | "arm64ec" => AARCH64_FEATURES,
            "x86" | "x86_64" => X86_FEATURES,
            "hexagon" => HEXAGON_FEATURES,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_FEATURES,
            "powerpc" | "powerpc64" => POWERPC_FEATURES,
            "riscv32" | "riscv64" => RISCV_FEATURES,
            "wasm32" | "wasm64" => WASM_FEATURES,
            "bpf" => BPF_FEATURES,
            "csky" => CSKY_FEATURES,
            "loongarch64" => LOONGARCH_FEATURES,
            "s390x" => IBMZ_FEATURES,
            "sparc" | "sparc64" => SPARC_FEATURES,
            _ => &[],
        }
    }

    pub fn features_for_correct_vector_abi(&self) -> &'static [(u64, &'static str)] {
        match &*self.arch {
            "x86" | "x86_64" => X86_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "aarch64" | "arm64ec" => AARCH64_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "arm" => ARM_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "powerpc" | "powerpc64" => POWERPC_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "loongarch64" => &[], // on-stack ABI, so we complain about all by-val vectors
            "riscv32" | "riscv64" => RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "wasm32" | "wasm64" => WASM_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "s390x" => S390X_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "sparc" | "sparc64" => SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "hexagon" => HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "bpf" => &[], // no vector ABI
            "csky" => CSKY_FEATURES_FOR_CORRECT_VECTOR_ABI,
            // FIXME: for some tier3 targets, we are overly cautious and always give warnings
            // when passing args in vector registers.
            _ => &[],
        }
    }

    pub fn tied_target_features(&self) -> &'static [&'static [&'static str]] {
        match &*self.arch {
            "aarch64" | "arm64ec" => AARCH64_TIED_FEATURES,
            _ => &[],
        }
    }

    pub fn implied_target_features(
        &self,
        base_features: impl Iterator<Item = Symbol>,
    ) -> FxHashSet<Symbol> {
        let implied_features = self
            .rust_target_features()
            .iter()
            .map(|(f, _, i)| (Symbol::intern(f), i))
            .collect::<FxHashMap<_, _>>();

        // implied target features have their own implied target features, so we traverse the
        // map until there are no more features to add
        let mut features = FxHashSet::default();
        let mut new_features = base_features.collect::<Vec<Symbol>>();
        while let Some(new_feature) = new_features.pop() {
            if features.insert(new_feature) {
                if let Some(implied_features) = implied_features.get(&new_feature) {
                    new_features.extend(implied_features.iter().copied().map(Symbol::intern))
                }
            }
        }
        features
    }
}
