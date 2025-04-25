//! Declares Rust's target feature names for each target.
//! Note that these are similar to but not always identical to LLVM's feature names,
//! and Rust adds some features that do not correspond to LLVM features at all.
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_span::{Symbol, sym};

use crate::spec::{FloatAbi, RustcAbi, Target};

/// Features that control behaviour of rustc, rather than the codegen.
/// These exist globally and are not in the target-specific lists below.
pub const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

/// Features that require special handling when passing to LLVM:
/// these are target-specific (i.e., must also be listed in the target-specific list below)
/// but do not correspond to an LLVM target feature.
pub const RUSTC_SPECIAL_FEATURES: &[&str] = &["backchain"];

/// Stability information for target features.
#[derive(Debug, Copy, Clone)]
pub enum Stability {
    /// This target feature is stable, it can be used in `#[target_feature]` and
    /// `#[cfg(target_feature)]`.
    Stable,
    /// This target feature is unstable. It is only present in `#[cfg(target_feature)]` on
    /// nightly and using it in `#[target_feature]` requires enabling the given nightly feature.
    Unstable(
        /// This must be a *language* feature, or else rustc will ICE when reporting a missing
        /// feature gate!
        Symbol,
    ),
    /// This feature can not be set via `-Ctarget-feature` or `#[target_feature]`, it can only be
    /// set in the target spec. It is never set in `cfg(target_feature)`. Used in
    /// particular for features are actually ABI configuration flags (not all targets are as nice as
    /// RISC-V and have an explicit way to set the ABI separate from target features).
    Forbidden { reason: &'static str },
}
use Stability::*;

impl<CTX> HashStable<CTX> for Stability {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Stability::Stable => {}
            Stability::Unstable(nightly_feature) => {
                nightly_feature.hash_stable(hcx, hasher);
            }
            Stability::Forbidden { reason } => {
                reason.hash_stable(hcx, hasher);
            }
        }
    }
}

impl Stability {
    /// Returns whether the feature can be used in `cfg(target_feature)` ever.
    /// (It might still be nightly-only even if this returns `true`, so make sure to also check
    /// `requires_nightly`.)
    pub fn in_cfg(&self) -> bool {
        !matches!(self, Stability::Forbidden { .. })
    }

    /// Returns the nightly feature that is required to toggle this target feature via
    /// `#[target_feature]`/`-Ctarget-feature` or to test it via `cfg(target_feature)`.
    /// (For `cfg` we only care whether the feature is nightly or not, we don't require
    /// the feature gate to actually be enabled when using a nightly compiler.)
    ///
    /// Before calling this, ensure the feature is even permitted for this use:
    /// - for `#[target_feature]`/`-Ctarget-feature`, check `allow_toggle()`
    /// - for `cfg(target_feature)`, check `in_cfg`
    pub fn requires_nightly(&self) -> Option<Symbol> {
        match *self {
            Stability::Unstable(nightly_feature) => Some(nightly_feature),
            Stability::Stable { .. } => None,
            Stability::Forbidden { .. } => panic!("forbidden features should not reach this far"),
        }
    }

    /// Returns whether the feature may be toggled via `#[target_feature]` or `-Ctarget-feature`.
    /// (It might still be nightly-only even if this returns `true`, so make sure to also check
    /// `requires_nightly`.)
    pub fn toggle_allowed(&self) -> Result<(), &'static str> {
        match self {
            Stability::Forbidden { reason } => Err(reason),
            _ => Ok(()),
        }
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
// It is important for soundness to consider the interaction of targets features and the function
// call ABI. For example, disabling the `x87` feature on x86 changes how scalar floats are passed as
// arguments, so letting people toggle that feature would be unsound. To this end, the
// `abi_required_features` function computes which target features must and must not be enabled for
// any given target, and individual features can also be marked as `Forbidden`.
// See https://github.com/rust-lang/rust/issues/116344 for some more context.
//
// The one exception to features that change the ABI is features that enable larger vector
// registers. Those are permitted to be listed here. The `*_FOR_CORRECT_VECTOR_ABI` arrays store
// information about which target feature is ABI-required for which vector size; this is used to
// ensure that vectors can only be passed via `extern "C"` when the right feature is enabled. (For
// the "Rust" ABI we generally pass vectors by-ref exactly to avoid these issues.)
// Also see https://github.com/rust-lang/rust/issues/116558.
//
// Stabilizing a target feature requires t-lang approval.

// If feature A "implies" feature B, then:
// - when A gets enabled (via `-Ctarget-feature` or `#[target_feature]`), we also enable B
// - when B gets disabled (via `-Ctarget-feature`), we also disable A
//
// Both of these are also applied transitively.
type ImpliedFeatures = &'static [&'static str];

static ARM_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("aclass", Unstable(sym::arm_target_feature), &[]),
    ("aes", Unstable(sym::arm_target_feature), &["neon"]),
    (
        "atomics-32",
        Stability::Forbidden { reason: "unsound because it changes the ABI of atomic operations" },
        &[],
    ),
    ("crc", Unstable(sym::arm_target_feature), &[]),
    ("d32", Unstable(sym::arm_target_feature), &[]),
    ("dotprod", Unstable(sym::arm_target_feature), &["neon"]),
    ("dsp", Unstable(sym::arm_target_feature), &[]),
    ("fp-armv8", Unstable(sym::arm_target_feature), &["vfp4"]),
    ("fp16", Unstable(sym::arm_target_feature), &["neon"]),
    ("fpregs", Unstable(sym::arm_target_feature), &[]),
    ("i8mm", Unstable(sym::arm_target_feature), &["neon"]),
    ("mclass", Unstable(sym::arm_target_feature), &[]),
    ("neon", Unstable(sym::arm_target_feature), &["vfp3"]),
    ("rclass", Unstable(sym::arm_target_feature), &[]),
    ("sha2", Unstable(sym::arm_target_feature), &["neon"]),
    // This can be *disabled* on non-`hf` targets to enable the use
    // of hardfloats while keeping the softfloat ABI.
    // FIXME before stabilization: Should we expose this as a `hard-float` target feature instead of
    // matching the odd negative feature LLVM uses?
    ("soft-float", Unstable(sym::arm_target_feature), &[]),
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
];

static AARCH64_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
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
    // We forbid directly toggling just `fp-armv8`; it must be toggled with `neon`.
    ("fp-armv8", Stability::Forbidden { reason: "Rust ties `fp-armv8` to `neon`" }, &[]),
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
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled globally using -Zfixed-x18, not
    // #[target_feature].
    // Note that cfg(target_feature = "reserve-x18") is currently not set for
    // targets that reserve x18 by default.
    ("reserve-x18", Unstable(sym::aarch64_unstable_target_feature), &[]),
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
    (
        "v8.1a",
        Unstable(sym::aarch64_ver_target_feature),
        &["crc", "lse", "rdm", "pan", "lor", "vh"],
    ),
    ("v8.2a", Unstable(sym::aarch64_ver_target_feature), &["v8.1a", "ras", "dpb"]),
    (
        "v8.3a",
        Unstable(sym::aarch64_ver_target_feature),
        &["v8.2a", "rcpc", "paca", "pacg", "jsconv"],
    ),
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

static X86_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("adx", Stable, &[]),
    ("aes", Stable, &["sse2"]),
    ("amx-avx512", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-bf16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-complex", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-fp16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-fp8", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-int8", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-movrs", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-tf32", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-tile", Unstable(sym::x86_amx_intrinsics), &[]),
    ("amx-transpose", Unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
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
    ("kl", Unstable(sym::keylocker_x86), &["sse2"]),
    ("lahfsahf", Unstable(sym::lahfsahf_target_feature), &[]),
    ("lzcnt", Stable, &[]),
    ("movbe", Stable, &[]),
    ("movrs", Unstable(sym::movrs_target_feature), &[]),
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
    // This cannot actually be toggled, the ABI always fixes it, so it'd make little sense to
    // stabilize. It must be in this list for the ABI check to be able to use it.
    ("soft-float", Stability::Unstable(sym::x87_target_feature), &[]),
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
    ("widekl", Unstable(sym::keylocker_x86), &["kl"]),
    ("x87", Unstable(sym::x87_target_feature), &[]),
    ("xop", Unstable(sym::xop_target_feature), &[/*"fma4", */ "avx", "sse4a"]),
    ("xsave", Stable, &[]),
    ("xsavec", Stable, &["xsave"]),
    ("xsaveopt", Stable, &["xsave"]),
    ("xsaves", Stable, &["xsave"]),
    // tidy-alphabetical-end
];

const HEXAGON_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("hvx", Unstable(sym::hexagon_target_feature), &[]),
    ("hvx-length128b", Unstable(sym::hexagon_target_feature), &["hvx"]),
    // tidy-alphabetical-end
];

static POWERPC_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("altivec", Unstable(sym::powerpc_target_feature), &[]),
    ("msync", Unstable(sym::powerpc_target_feature), &[]),
    ("partword-atomics", Unstable(sym::powerpc_target_feature), &[]),
    ("power10-vector", Unstable(sym::powerpc_target_feature), &["power9-vector"]),
    ("power8-altivec", Unstable(sym::powerpc_target_feature), &["altivec"]),
    ("power8-crypto", Unstable(sym::powerpc_target_feature), &["power8-altivec"]),
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

static RISCV_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("a", Stable, &["zaamo", "zalrsc"]),
    ("b", Unstable(sym::riscv_target_feature), &["zba", "zbb", "zbs"]),
    ("c", Stable, &["zca"]),
    ("d", Unstable(sym::riscv_target_feature), &["f"]),
    ("e", Unstable(sym::riscv_target_feature), &[]),
    ("f", Unstable(sym::riscv_target_feature), &["zicsr"]),
    (
        "forced-atomics",
        Stability::Forbidden { reason: "unsound because it changes the ABI of atomic operations" },
        &[],
    ),
    ("m", Stable, &[]),
    ("relax", Unstable(sym::riscv_target_feature), &[]),
    ("unaligned-scalar-mem", Unstable(sym::riscv_target_feature), &[]),
    ("unaligned-vector-mem", Unstable(sym::riscv_target_feature), &[]),
    ("v", Unstable(sym::riscv_target_feature), &["zvl128b", "zve64d"]),
    ("za128rs", Unstable(sym::riscv_target_feature), &[]),
    ("za64rs", Unstable(sym::riscv_target_feature), &[]),
    ("zaamo", Unstable(sym::riscv_target_feature), &[]),
    ("zabha", Unstable(sym::riscv_target_feature), &["zaamo"]),
    ("zacas", Unstable(sym::riscv_target_feature), &["zaamo"]),
    ("zalrsc", Unstable(sym::riscv_target_feature), &[]),
    ("zama16b", Unstable(sym::riscv_target_feature), &[]),
    ("zawrs", Unstable(sym::riscv_target_feature), &[]),
    ("zba", Stable, &[]),
    ("zbb", Stable, &[]),
    ("zbc", Stable, &["zbkc"]), // Zbc ⊃ Zbkc
    ("zbkb", Stable, &[]),
    ("zbkc", Stable, &[]),
    ("zbkx", Stable, &[]),
    ("zbs", Stable, &[]),
    ("zca", Unstable(sym::riscv_target_feature), &[]),
    ("zcb", Unstable(sym::riscv_target_feature), &["zca"]),
    ("zcmop", Unstable(sym::riscv_target_feature), &["zca"]),
    ("zdinx", Unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zfa", Unstable(sym::riscv_target_feature), &["f"]),
    ("zfh", Unstable(sym::riscv_target_feature), &["zfhmin"]),
    ("zfhmin", Unstable(sym::riscv_target_feature), &["f"]),
    ("zfinx", Unstable(sym::riscv_target_feature), &["zicsr"]),
    ("zhinx", Unstable(sym::riscv_target_feature), &["zhinxmin"]),
    ("zhinxmin", Unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zicboz", Unstable(sym::riscv_target_feature), &[]),
    ("zicntr", Unstable(sym::riscv_target_feature), &["zicsr"]),
    ("zicond", Unstable(sym::riscv_target_feature), &[]),
    ("zicsr", Unstable(sym::riscv_target_feature), &[]),
    ("zifencei", Unstable(sym::riscv_target_feature), &[]),
    ("zihintntl", Unstable(sym::riscv_target_feature), &[]),
    ("zihintpause", Unstable(sym::riscv_target_feature), &[]),
    ("zihpm", Unstable(sym::riscv_target_feature), &["zicsr"]),
    ("zimop", Unstable(sym::riscv_target_feature), &[]),
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
    ("ztso", Unstable(sym::riscv_target_feature), &[]),
    ("zvbb", Unstable(sym::riscv_target_feature), &["zvkb"]), // Zvbb ⊃ Zvkb
    ("zvbc", Unstable(sym::riscv_target_feature), &["zve64x"]),
    ("zve32f", Unstable(sym::riscv_target_feature), &["zve32x", "f"]),
    ("zve32x", Unstable(sym::riscv_target_feature), &["zvl32b", "zicsr"]),
    ("zve64d", Unstable(sym::riscv_target_feature), &["zve64f", "d"]),
    ("zve64f", Unstable(sym::riscv_target_feature), &["zve32f", "zve64x"]),
    ("zve64x", Unstable(sym::riscv_target_feature), &["zve32x", "zvl64b"]),
    ("zvfh", Unstable(sym::riscv_target_feature), &["zvfhmin", "zve32f", "zfhmin"]), // Zvfh ⊃ Zvfhmin
    ("zvfhmin", Unstable(sym::riscv_target_feature), &["zve32f"]),
    ("zvkb", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvkg", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvkn", Unstable(sym::riscv_target_feature), &["zvkned", "zvknhb", "zvkb", "zvkt"]),
    ("zvknc", Unstable(sym::riscv_target_feature), &["zvkn", "zvbc"]),
    ("zvkned", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvkng", Unstable(sym::riscv_target_feature), &["zvkn", "zvkg"]),
    ("zvknha", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvknhb", Unstable(sym::riscv_target_feature), &["zvknha", "zve64x"]), // Zvknhb ⊃ Zvknha
    ("zvks", Unstable(sym::riscv_target_feature), &["zvksed", "zvksh", "zvkb", "zvkt"]),
    ("zvksc", Unstable(sym::riscv_target_feature), &["zvks", "zvbc"]),
    ("zvksed", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvksg", Unstable(sym::riscv_target_feature), &["zvks", "zvkg"]),
    ("zvksh", Unstable(sym::riscv_target_feature), &["zve32x"]),
    ("zvkt", Unstable(sym::riscv_target_feature), &[]),
    ("zvl1024b", Unstable(sym::riscv_target_feature), &["zvl512b"]),
    ("zvl128b", Unstable(sym::riscv_target_feature), &["zvl64b"]),
    ("zvl16384b", Unstable(sym::riscv_target_feature), &["zvl8192b"]),
    ("zvl2048b", Unstable(sym::riscv_target_feature), &["zvl1024b"]),
    ("zvl256b", Unstable(sym::riscv_target_feature), &["zvl128b"]),
    ("zvl32768b", Unstable(sym::riscv_target_feature), &["zvl16384b"]),
    ("zvl32b", Unstable(sym::riscv_target_feature), &[]),
    ("zvl4096b", Unstable(sym::riscv_target_feature), &["zvl2048b"]),
    ("zvl512b", Unstable(sym::riscv_target_feature), &["zvl256b"]),
    ("zvl64b", Unstable(sym::riscv_target_feature), &["zvl32b"]),
    ("zvl65536b", Unstable(sym::riscv_target_feature), &["zvl32768b"]),
    ("zvl8192b", Unstable(sym::riscv_target_feature), &["zvl4096b"]),
    // tidy-alphabetical-end
];

static WASM_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
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

static CSKY_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
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

static LOONGARCH_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("d", Unstable(sym::loongarch_target_feature), &["f"]),
    ("div32", Unstable(sym::loongarch_target_feature), &[]),
    ("f", Unstable(sym::loongarch_target_feature), &[]),
    ("frecipe", Unstable(sym::loongarch_target_feature), &[]),
    ("lam-bh", Unstable(sym::loongarch_target_feature), &[]),
    ("lamcas", Unstable(sym::loongarch_target_feature), &[]),
    ("lasx", Unstable(sym::loongarch_target_feature), &["lsx"]),
    ("lbt", Unstable(sym::loongarch_target_feature), &[]),
    ("ld-seq-sa", Unstable(sym::loongarch_target_feature), &[]),
    ("lsx", Unstable(sym::loongarch_target_feature), &["d"]),
    ("lvz", Unstable(sym::loongarch_target_feature), &[]),
    ("relax", Unstable(sym::loongarch_target_feature), &[]),
    ("scq", Unstable(sym::loongarch_target_feature), &[]),
    ("ual", Unstable(sym::loongarch_target_feature), &[]),
    // tidy-alphabetical-end
];

const IBMZ_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("backchain", Unstable(sym::s390x_target_feature), &[]),
    ("deflate-conversion", Unstable(sym::s390x_target_feature), &[]),
    ("enhanced-sort", Unstable(sym::s390x_target_feature), &[]),
    ("guarded-storage", Unstable(sym::s390x_target_feature), &[]),
    ("high-word", Unstable(sym::s390x_target_feature), &[]),
    ("nnp-assist", Unstable(sym::s390x_target_feature), &["vector"]),
    ("transactional-execution", Unstable(sym::s390x_target_feature), &[]),
    ("vector", Unstable(sym::s390x_target_feature), &[]),
    ("vector-enhancements-1", Unstable(sym::s390x_target_feature), &["vector"]),
    ("vector-enhancements-2", Unstable(sym::s390x_target_feature), &["vector-enhancements-1"]),
    ("vector-packed-decimal", Unstable(sym::s390x_target_feature), &["vector"]),
    (
        "vector-packed-decimal-enhancement",
        Unstable(sym::s390x_target_feature),
        &["vector-packed-decimal"],
    ),
    (
        "vector-packed-decimal-enhancement-2",
        Unstable(sym::s390x_target_feature),
        &["vector-packed-decimal-enhancement"],
    ),
    // tidy-alphabetical-end
];

const SPARC_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("leoncasa", Unstable(sym::sparc_target_feature), &[]),
    ("v8plus", Unstable(sym::sparc_target_feature), &[]),
    ("v9", Unstable(sym::sparc_target_feature), &[]),
    // tidy-alphabetical-end
];

static M68K_FEATURES: &[(&str, Stability, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("isa-68000", Unstable(sym::m68k_target_feature), &[]),
    ("isa-68010", Unstable(sym::m68k_target_feature), &["isa-68000"]),
    ("isa-68020", Unstable(sym::m68k_target_feature), &["isa-68010"]),
    ("isa-68030", Unstable(sym::m68k_target_feature), &["isa-68020"]),
    ("isa-68040", Unstable(sym::m68k_target_feature), &["isa-68030", "isa-68882"]),
    ("isa-68060", Unstable(sym::m68k_target_feature), &["isa-68040"]),
    // FPU
    ("isa-68881", Unstable(sym::m68k_target_feature), &[]),
    ("isa-68882", Unstable(sym::m68k_target_feature), &["isa-68881"]),
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
        .chain(M68K_FEATURES)
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
const RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[
    (32, "zvl32b"),
    (64, "zvl64b"),
    (128, "zvl128b"),
    (256, "zvl256b"),
    (512, "zvl512b"),
    (1024, "zvl1024b"),
    (2048, "zvl2048b"),
    (4096, "zvl4096b"),
    (8192, "zvl8192b"),
    (16384, "zvl16384b"),
    (32768, "zvl32768b"),
    (65536, "zvl65536b"),
];
// Always error on SPARC, as the necessary target features cannot be enabled in Rust at the moment.
const SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[/*(64, "vis")*/];

const HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[/*(512, "hvx-length64b"),*/ (1024, "hvx-length128b")];
const MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "msa")];
const CSKY_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "vdspv1")];
const LOONGARCH_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[(128, "lsx"), (256, "lasx")];

#[derive(Copy, Clone, Debug)]
pub struct FeatureConstraints {
    /// Features that must be enabled.
    pub required: &'static [&'static str],
    /// Features that must be disabled.
    pub incompatible: &'static [&'static str],
}

impl Target {
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
            "m68k" => M68K_FEATURES,
            _ => &[],
        }
    }

    pub fn features_for_correct_vector_abi(&self) -> &'static [(u64, &'static str)] {
        match &*self.arch {
            "x86" | "x86_64" => X86_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "aarch64" | "arm64ec" => AARCH64_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "arm" => ARM_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "powerpc" | "powerpc64" => POWERPC_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "loongarch64" => LOONGARCH_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "riscv32" | "riscv64" => RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "wasm32" | "wasm64" => WASM_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "s390x" => S390X_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "sparc" | "sparc64" => SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "hexagon" => HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "bpf" | "m68k" => &[], // no vector ABI
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

    // Note: the returned set includes `base_feature`.
    pub fn implied_target_features<'a>(&self, base_feature: &'a str) -> FxHashSet<&'a str> {
        let implied_features =
            self.rust_target_features().iter().map(|(f, _, i)| (f, i)).collect::<FxHashMap<_, _>>();

        // Implied target features have their own implied target features, so we traverse the
        // map until there are no more features to add.
        let mut features = FxHashSet::default();
        let mut new_features = vec![base_feature];
        while let Some(new_feature) = new_features.pop() {
            if features.insert(new_feature) {
                if let Some(implied_features) = implied_features.get(&new_feature) {
                    new_features.extend(implied_features.iter().copied())
                }
            }
        }
        features
    }

    /// Returns two lists of features:
    /// the first list contains target features that must be enabled for ABI reasons,
    /// and the second list contains target feature that must be disabled for ABI reasons.
    ///
    /// These features are automatically appended to whatever the target spec sets as default
    /// features for the target.
    ///
    /// All features enabled/disabled via `-Ctarget-features` and `#[target_features]` are checked
    /// against this. We also check any implied features, based on the information above. If LLVM
    /// implicitly enables more implied features than we do, that could bypass this check!
    pub fn abi_required_features(&self) -> FeatureConstraints {
        const NOTHING: FeatureConstraints = FeatureConstraints { required: &[], incompatible: &[] };
        // Some architectures don't have a clean explicit ABI designation; instead, the ABI is
        // defined by target features. When that is the case, those target features must be
        // "forbidden" in the list above to ensure that there is a consistent answer to the
        // questions "which ABI is used".
        match &*self.arch {
            "x86" => {
                // We use our own ABI indicator here; LLVM does not have anything native.
                // Every case should require or forbid `soft-float`!
                match self.rustc_abi {
                    None => {
                        // Default hardfloat ABI.
                        // x87 must be enabled, soft-float must be disabled.
                        FeatureConstraints { required: &["x87"], incompatible: &["soft-float"] }
                    }
                    Some(RustcAbi::X86Sse2) => {
                        // Extended hardfloat ABI. x87 and SSE2 must be enabled, soft-float must be disabled.
                        FeatureConstraints {
                            required: &["x87", "sse2"],
                            incompatible: &["soft-float"],
                        }
                    }
                    Some(RustcAbi::X86Softfloat) => {
                        // Softfloat ABI, requires corresponding target feature. That feature trumps
                        // `x87` and all other FPU features so those do not matter.
                        // Note that this one requirement is the entire implementation of the ABI!
                        // LLVM handles the rest.
                        FeatureConstraints { required: &["soft-float"], incompatible: &[] }
                    }
                }
            }
            "x86_64" => {
                // We use our own ABI indicator here; LLVM does not have anything native.
                // Every case should require or forbid `soft-float`!
                match self.rustc_abi {
                    None => {
                        // Default hardfloat ABI. On x86-64, this always includes SSE2.
                        FeatureConstraints {
                            required: &["x87", "sse2"],
                            incompatible: &["soft-float"],
                        }
                    }
                    Some(RustcAbi::X86Softfloat) => {
                        // Softfloat ABI, requires corresponding target feature. That feature trumps
                        // `x87` and all other FPU features so those do not matter.
                        // Note that this one requirement is the entire implementation of the ABI!
                        // LLVM handles the rest.
                        FeatureConstraints { required: &["soft-float"], incompatible: &[] }
                    }
                    Some(r) => panic!("invalid Rust ABI for x86_64: {r:?}"),
                }
            }
            "arm" => {
                // On ARM, ABI handling is reasonably sane; we use `llvm_floatabi` to indicate
                // to LLVM which ABI we are going for.
                match self.llvm_floatabi.unwrap() {
                    FloatAbi::Soft => {
                        // Nothing special required, will use soft-float ABI throughout.
                        // We can even allow `-soft-float` here; in fact that is useful as it lets
                        // people use FPU instructions with a softfloat ABI (corresponds to
                        // `-mfloat-abi=softfp` in GCC/clang).
                        NOTHING
                    }
                    FloatAbi::Hard => {
                        // Must have `fpregs` and must not have `soft-float`.
                        FeatureConstraints { required: &["fpregs"], incompatible: &["soft-float"] }
                    }
                }
            }
            "aarch64" | "arm64ec" => {
                // Aarch64 has no sane ABI specifier, and LLVM doesn't even have a way to force
                // the use of soft-float, so all we can do here is some crude hacks.
                match &*self.abi {
                    "softfloat" => {
                        // This is not fully correct, LLVM actually doesn't let us enforce the softfloat
                        // ABI properly... see <https://github.com/rust-lang/rust/issues/134375>.
                        // FIXME: should we forbid "neon" here? But that would be a breaking change.
                        NOTHING
                    }
                    _ => {
                        // Everything else is assumed to use a hardfloat ABI. neon and fp-armv8 must be enabled.
                        // These are Rust feature names and we use "neon" to control both of them.
                        FeatureConstraints { required: &["neon"], incompatible: &[] }
                    }
                }
            }
            "riscv32" | "riscv64" => {
                // RISC-V handles ABI in a very sane way, being fully explicit via `llvm_abiname`
                // about what the intended ABI is.
                match &*self.llvm_abiname {
                    "ilp32d" | "lp64d" => {
                        // Requires d (which implies f), incompatible with e.
                        FeatureConstraints { required: &["d"], incompatible: &["e"] }
                    }
                    "ilp32f" | "lp64f" => {
                        // Requires f, incompatible with e.
                        FeatureConstraints { required: &["f"], incompatible: &["e"] }
                    }
                    "ilp32" | "lp64" => {
                        // Requires nothing, incompatible with e.
                        FeatureConstraints { required: &[], incompatible: &["e"] }
                    }
                    "ilp32e" => {
                        // ilp32e is documented to be incompatible with features that need aligned
                        // load/stores > 32 bits, like `d`. (One could also just generate more
                        // complicated code to align the stack when needed, but the RISCV
                        // architecture manual just explicitly rules out this combination so we
                        // might as well.)
                        // Note that the `e` feature is not required: the ABI treats the extra
                        // registers as caller-save, so it is safe to use them only in some parts of
                        // a program while the rest doesn't know they even exist.
                        FeatureConstraints { required: &[], incompatible: &["d"] }
                    }
                    "lp64e" => {
                        // As above, `e` is not required.
                        NOTHING
                    }
                    _ => unreachable!(),
                }
            }
            "loongarch64" => {
                // LoongArch handles ABI in a very sane way, being fully explicit via `llvm_abiname`
                // about what the intended ABI is.
                match &*self.llvm_abiname {
                    "ilp32d" | "lp64d" => {
                        // Requires d (which implies f), incompatible with nothing.
                        FeatureConstraints { required: &["d"], incompatible: &[] }
                    }
                    "ilp32f" | "lp64f" => {
                        // Requires f, incompatible with nothing.
                        FeatureConstraints { required: &["f"], incompatible: &[] }
                    }
                    "ilp32s" | "lp64s" => {
                        // The soft-float ABI does not require any features and is also not
                        // incompatible with any features. Rust targets explicitly specify the
                        // LLVM ABI names, which allows for enabling hard-float support even on
                        // soft-float targets, and ensures that the ABI behavior is as expected.
                        NOTHING
                    }
                    _ => unreachable!(),
                }
            }
            _ => NOTHING,
        }
    }
}
