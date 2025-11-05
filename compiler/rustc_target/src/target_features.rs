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
        matches!(self, Stability::Stable | Stability::Unstable { .. })
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
            Stability::Unstable(_) | Stability::Stable { .. } => Ok(()),
            Stability::Forbidden { reason } => Err(reason),
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
// Additionally, if the feature is not available in older version of LLVM supported by the current
// rust, the same function must be updated to filter out these features to avoid triggering
// warnings.
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

// If feature is a target modifier.
type TargetModifier = bool;

static ARM_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("aclass", Unstable(sym::arm_target_feature), &[], false),
    ("aes", Unstable(sym::arm_target_feature), &["neon"], false),
    (
        "atomics-32",
        Stability::Forbidden { reason: "unsound because it changes the ABI of atomic operations" },
        &[],
        false,
    ),
    ("crc", Unstable(sym::arm_target_feature), &[], false),
    ("d32", Unstable(sym::arm_target_feature), &[], false),
    ("dotprod", Unstable(sym::arm_target_feature), &["neon"], false),
    ("dsp", Unstable(sym::arm_target_feature), &[], false),
    ("fp-armv8", Unstable(sym::arm_target_feature), &["vfp4"], false),
    ("fp16", Unstable(sym::arm_target_feature), &["neon"], false),
    ("fpregs", Unstable(sym::arm_target_feature), &[], false),
    ("i8mm", Unstable(sym::arm_target_feature), &["neon"], false),
    ("mclass", Unstable(sym::arm_target_feature), &[], false),
    ("neon", Unstable(sym::arm_target_feature), &["vfp3"], false),
    ("rclass", Unstable(sym::arm_target_feature), &[], false),
    ("sha2", Unstable(sym::arm_target_feature), &["neon"], false),
    // This can be *disabled* on non-`hf` targets to enable the use
    // of hardfloats while keeping the softfloat ABI.
    // FIXME before stabilization: Should we expose this as a `hard-float` target feature instead of
    // matching the odd negative feature LLVM uses?
    ("soft-float", Unstable(sym::arm_target_feature), &[], false),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", Unstable(sym::arm_target_feature), &[], false),
    ("thumb2", Unstable(sym::arm_target_feature), &[], false),
    ("trustzone", Unstable(sym::arm_target_feature), &[], false),
    ("v5te", Unstable(sym::arm_target_feature), &[], false),
    ("v6", Unstable(sym::arm_target_feature), &["v5te"], false),
    ("v6k", Unstable(sym::arm_target_feature), &["v6"], false),
    ("v6t2", Unstable(sym::arm_target_feature), &["v6k", "thumb2"], false),
    ("v7", Unstable(sym::arm_target_feature), &["v6t2"], false),
    ("v8", Unstable(sym::arm_target_feature), &["v7"], false),
    ("vfp2", Unstable(sym::arm_target_feature), &[], false),
    ("vfp3", Unstable(sym::arm_target_feature), &["vfp2", "d32"], false),
    ("vfp4", Unstable(sym::arm_target_feature), &["vfp3"], false),
    ("virtualization", Unstable(sym::arm_target_feature), &[], false),
    // tidy-alphabetical-end
];

static AARCH64_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    // FEAT_AES & FEAT_PMULL
    ("aes", Stable, &["neon"], false),
    // FEAT_BF16
    ("bf16", Stable, &[], false),
    // FEAT_BTI
    ("bti", Stable, &[], false),
    // FEAT_CRC
    ("crc", Stable, &[], false),
    // FEAT_CSSC
    ("cssc", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_DIT
    ("dit", Stable, &[], false),
    // FEAT_DotProd
    ("dotprod", Stable, &["neon"], false),
    // FEAT_DPB
    ("dpb", Stable, &[], false),
    // FEAT_DPB2
    ("dpb2", Stable, &["dpb"], false),
    // FEAT_ECV
    ("ecv", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_F32MM
    ("f32mm", Stable, &["sve"], false),
    // FEAT_F64MM
    ("f64mm", Stable, &["sve"], false),
    // FEAT_FAMINMAX
    ("faminmax", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_FCMA
    ("fcma", Stable, &["neon"], false),
    // FEAT_FHM
    ("fhm", Stable, &["fp16"], false),
    // FEAT_FLAGM
    ("flagm", Stable, &[], false),
    // FEAT_FLAGM2
    ("flagm2", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // We forbid directly toggling just `fp-armv8`; it must be toggled with `neon`.
    ("fp-armv8", Stability::Forbidden { reason: "Rust ties `fp-armv8` to `neon`" }, &[], false),
    // FEAT_FP8
    ("fp8", Unstable(sym::aarch64_unstable_target_feature), &["faminmax", "lut", "bf16"], false),
    // FEAT_FP8DOT2
    ("fp8dot2", Unstable(sym::aarch64_unstable_target_feature), &["fp8dot4"], false),
    // FEAT_FP8DOT4
    ("fp8dot4", Unstable(sym::aarch64_unstable_target_feature), &["fp8fma"], false),
    // FEAT_FP8FMA
    ("fp8fma", Unstable(sym::aarch64_unstable_target_feature), &["fp8"], false),
    // FEAT_FP16
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("fp16", Stable, &["neon"], false),
    // FEAT_FRINTTS
    ("frintts", Stable, &[], false),
    // FEAT_HBC
    ("hbc", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_I8MM
    ("i8mm", Stable, &[], false),
    // FEAT_JSCVT
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("jsconv", Stable, &["neon"], false),
    // FEAT_LOR
    ("lor", Stable, &[], false),
    // FEAT_LSE
    ("lse", Stable, &[], false),
    // FEAT_LSE2
    ("lse2", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_LSE128
    ("lse128", Unstable(sym::aarch64_unstable_target_feature), &["lse"], false),
    // FEAT_LUT
    ("lut", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_MOPS
    ("mops", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_MTE & FEAT_MTE2
    ("mte", Stable, &[], false),
    // FEAT_AdvSimd & FEAT_FP
    ("neon", Stable, &[], false),
    // Backend option to turn atomic operations into an intrinsic call when `lse` is not known to be
    // available, so the intrinsic can do runtime LSE feature detection rather than unconditionally
    // using slower non-LSE operations. Unstable since it doesn't need to user-togglable.
    ("outline-atomics", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_PAUTH (address authentication)
    ("paca", Stable, &[], false),
    // FEAT_PAUTH (generic authentication)
    ("pacg", Stable, &[], false),
    // FEAT_PAN
    ("pan", Stable, &[], false),
    // FEAT_PAuth_LR
    ("pauth-lr", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_PMUv3
    ("pmuv3", Stable, &[], false),
    // FEAT_RNG
    ("rand", Stable, &[], false),
    // FEAT_RAS & FEAT_RASv1p1
    ("ras", Stable, &[], false),
    // FEAT_LRCPC
    ("rcpc", Stable, &[], false),
    // FEAT_LRCPC2
    ("rcpc2", Stable, &["rcpc"], false),
    // FEAT_LRCPC3
    ("rcpc3", Unstable(sym::aarch64_unstable_target_feature), &["rcpc2"], false),
    // FEAT_RDM
    ("rdm", Stable, &["neon"], false),
    ("reserve-x18", Forbidden { reason: "use `-Zfixed-x18` compiler flag instead" }, &[], false),
    // FEAT_SB
    ("sb", Stable, &[], false),
    // FEAT_SHA1 & FEAT_SHA256
    ("sha2", Stable, &["neon"], false),
    // FEAT_SHA512 & FEAT_SHA3
    ("sha3", Stable, &["sha2"], false),
    // FEAT_SM3 & FEAT_SM4
    ("sm4", Stable, &["neon"], false),
    // FEAT_SME
    ("sme", Unstable(sym::aarch64_unstable_target_feature), &["bf16"], false),
    // FEAT_SME_B16B16
    (
        "sme-b16b16",
        Unstable(sym::aarch64_unstable_target_feature),
        &["bf16", "sme2", "sve-b16b16"],
        false,
    ),
    // FEAT_SME_F8F16
    ("sme-f8f16", Unstable(sym::aarch64_unstable_target_feature), &["sme-f8f32"], false),
    // FEAT_SME_F8F32
    ("sme-f8f32", Unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"], false),
    // FEAT_SME_F16F16
    ("sme-f16f16", Unstable(sym::aarch64_unstable_target_feature), &["sme2"], false),
    // FEAT_SME_F64F64
    ("sme-f64f64", Unstable(sym::aarch64_unstable_target_feature), &["sme"], false),
    // FEAT_SME_FA64
    ("sme-fa64", Unstable(sym::aarch64_unstable_target_feature), &["sme", "sve2"], false),
    // FEAT_SME_I16I64
    ("sme-i16i64", Unstable(sym::aarch64_unstable_target_feature), &["sme"], false),
    // FEAT_SME_LUTv2
    ("sme-lutv2", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // FEAT_SME2
    ("sme2", Unstable(sym::aarch64_unstable_target_feature), &["sme"], false),
    // FEAT_SME2p1
    ("sme2p1", Unstable(sym::aarch64_unstable_target_feature), &["sme2"], false),
    // FEAT_SPE
    ("spe", Stable, &[], false),
    // FEAT_SSBS & FEAT_SSBS2
    ("ssbs", Stable, &[], false),
    // FEAT_SSVE_FP8FDOT2
    ("ssve-fp8dot2", Unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8dot4"], false),
    // FEAT_SSVE_FP8FDOT4
    ("ssve-fp8dot4", Unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8fma"], false),
    // FEAT_SSVE_FP8FMA
    ("ssve-fp8fma", Unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"], false),
    // FEAT_SVE
    // It was decided that SVE requires Neon: https://github.com/rust-lang/rust/pull/91608
    //
    // LLVM doesn't enable Neon for SVE. ARM indicates that they're separate, but probably always
    // exist together: https://developer.arm.com/documentation/102340/0100/New-features-in-SVE2
    //
    // "For backwards compatibility, Neon and VFP are required in the latest architectures."
    ("sve", Stable, &["neon"], false),
    // FEAT_SVE_B16B16 (SVE or SME Z-targeting instructions)
    ("sve-b16b16", Unstable(sym::aarch64_unstable_target_feature), &["bf16"], false),
    // FEAT_SVE2
    ("sve2", Stable, &["sve"], false),
    // FEAT_SVE_AES & FEAT_SVE_PMULL128
    ("sve2-aes", Stable, &["sve2", "aes"], false),
    // FEAT_SVE2_BitPerm
    ("sve2-bitperm", Stable, &["sve2"], false),
    // FEAT_SVE2_SHA3
    ("sve2-sha3", Stable, &["sve2", "sha3"], false),
    // FEAT_SVE2_SM4
    ("sve2-sm4", Stable, &["sve2", "sm4"], false),
    // FEAT_SVE2p1
    ("sve2p1", Unstable(sym::aarch64_unstable_target_feature), &["sve2"], false),
    // FEAT_TME
    ("tme", Stable, &[], false),
    (
        "v8.1a",
        Unstable(sym::aarch64_ver_target_feature),
        &["crc", "lse", "rdm", "pan", "lor", "vh"],
        false,
    ),
    ("v8.2a", Unstable(sym::aarch64_ver_target_feature), &["v8.1a", "ras", "dpb"], false),
    (
        "v8.3a",
        Unstable(sym::aarch64_ver_target_feature),
        &["v8.2a", "rcpc", "paca", "pacg", "jsconv"],
        false,
    ),
    (
        "v8.4a",
        Unstable(sym::aarch64_ver_target_feature),
        &["v8.3a", "dotprod", "dit", "flagm"],
        false,
    ),
    (
        "v8.5a",
        Unstable(sym::aarch64_ver_target_feature),
        &["v8.4a", "ssbs", "sb", "dpb2", "bti"],
        false,
    ),
    ("v8.6a", Unstable(sym::aarch64_ver_target_feature), &["v8.5a", "bf16", "i8mm"], false),
    ("v8.7a", Unstable(sym::aarch64_ver_target_feature), &["v8.6a", "wfxt"], false),
    ("v8.8a", Unstable(sym::aarch64_ver_target_feature), &["v8.7a", "hbc", "mops"], false),
    ("v8.9a", Unstable(sym::aarch64_ver_target_feature), &["v8.8a", "cssc"], false),
    ("v9.1a", Unstable(sym::aarch64_ver_target_feature), &["v9a", "v8.6a"], false),
    ("v9.2a", Unstable(sym::aarch64_ver_target_feature), &["v9.1a", "v8.7a"], false),
    ("v9.3a", Unstable(sym::aarch64_ver_target_feature), &["v9.2a", "v8.8a"], false),
    ("v9.4a", Unstable(sym::aarch64_ver_target_feature), &["v9.3a", "v8.9a"], false),
    ("v9.5a", Unstable(sym::aarch64_ver_target_feature), &["v9.4a"], false),
    ("v9a", Unstable(sym::aarch64_ver_target_feature), &["v8.5a", "sve2"], false),
    // FEAT_VHE
    ("vh", Stable, &[], false),
    // FEAT_WFxT
    ("wfxt", Unstable(sym::aarch64_unstable_target_feature), &[], false),
    // tidy-alphabetical-end
];

const AARCH64_TIED_FEATURES: &[&[&str]] = &[
    &["paca", "pacg"], // Together these represent `pauth` in LLVM
];

static X86_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("adx", Stable, &[], false),
    ("aes", Stable, &["sse2"], false),
    ("amx-avx512", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-bf16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-complex", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-fp8", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-fp16", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-int8", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-movrs", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-tf32", Unstable(sym::x86_amx_intrinsics), &["amx-tile"], false),
    ("amx-tile", Unstable(sym::x86_amx_intrinsics), &[], false),
    ("apxf", Unstable(sym::apx_target_feature), &[], false),
    ("avx", Stable, &["sse4.2"], false),
    ("avx2", Stable, &["avx"], false),
    (
        "avx10.1",
        Unstable(sym::avx10_target_feature),
        &[
            "avx512bf16",
            "avx512bitalg",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512f",
            "avx512fp16",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512vl",
            "avx512vnni",
            "avx512vpopcntdq",
        ],
        false,
    ),
    ("avx10.2", Unstable(sym::avx10_target_feature), &["avx10.1"], false),
    ("avx512bf16", Stable, &["avx512bw"], false),
    ("avx512bitalg", Stable, &["avx512bw"], false),
    ("avx512bw", Stable, &["avx512f"], false),
    ("avx512cd", Stable, &["avx512f"], false),
    ("avx512dq", Stable, &["avx512f"], false),
    ("avx512f", Stable, &["avx2", "fma", "f16c"], false),
    ("avx512fp16", Stable, &["avx512bw"], false),
    ("avx512ifma", Stable, &["avx512f"], false),
    ("avx512vbmi", Stable, &["avx512bw"], false),
    ("avx512vbmi2", Stable, &["avx512bw"], false),
    ("avx512vl", Stable, &["avx512f"], false),
    ("avx512vnni", Stable, &["avx512f"], false),
    ("avx512vp2intersect", Stable, &["avx512f"], false),
    ("avx512vpopcntdq", Stable, &["avx512f"], false),
    ("avxifma", Stable, &["avx2"], false),
    ("avxneconvert", Stable, &["avx2"], false),
    ("avxvnni", Stable, &["avx2"], false),
    ("avxvnniint8", Stable, &["avx2"], false),
    ("avxvnniint16", Stable, &["avx2"], false),
    ("bmi1", Stable, &[], false),
    ("bmi2", Stable, &[], false),
    ("cmpxchg16b", Stable, &[], false),
    ("ermsb", Unstable(sym::ermsb_target_feature), &[], false),
    ("f16c", Stable, &["avx"], false),
    ("fma", Stable, &["avx"], false),
    ("fxsr", Stable, &[], false),
    ("gfni", Stable, &["sse2"], false),
    ("kl", Stable, &["sse2"], false),
    ("lahfsahf", Unstable(sym::lahfsahf_target_feature), &[], false),
    ("lzcnt", Stable, &[], false),
    ("movbe", Stable, &[], false),
    ("movrs", Unstable(sym::movrs_target_feature), &[], false),
    ("pclmulqdq", Stable, &["sse2"], false),
    ("popcnt", Stable, &[], false),
    ("prfchw", Unstable(sym::prfchw_target_feature), &[], false),
    ("rdrand", Stable, &[], false),
    ("rdseed", Stable, &[], false),
    (
        "retpoline-external-thunk",
        Stability::Forbidden { reason: "use `-Zretpoline-external-thunk` compiler flag instead" },
        &[],
        false,
    ),
    (
        "retpoline-indirect-branches",
        Stability::Forbidden { reason: "use `-Zretpoline` compiler flag instead" },
        &[],
        false,
    ),
    (
        "retpoline-indirect-calls",
        Stability::Forbidden { reason: "use `-Zretpoline` compiler flag instead" },
        &[],
        false,
    ),
    ("rtm", Unstable(sym::rtm_target_feature), &[], false),
    ("sha", Stable, &["sse2"], false),
    ("sha512", Stable, &["avx2"], false),
    ("sm3", Stable, &["avx"], false),
    ("sm4", Stable, &["avx2"], false),
    // This cannot actually be toggled, the ABI always fixes it, so it'd make little sense to
    // stabilize. It must be in this list for the ABI check to be able to use it.
    ("soft-float", Stability::Unstable(sym::x87_target_feature), &[], false),
    ("sse", Stable, &[], false),
    ("sse2", Stable, &["sse"], false),
    ("sse3", Stable, &["sse2"], false),
    ("sse4.1", Stable, &["ssse3"], false),
    ("sse4.2", Stable, &["sse4.1"], false),
    ("sse4a", Stable, &["sse3"], false),
    ("ssse3", Stable, &["sse3"], false),
    ("tbm", Stable, &[], false),
    ("vaes", Stable, &["avx2", "aes"], false),
    ("vpclmulqdq", Stable, &["avx", "pclmulqdq"], false),
    ("widekl", Stable, &["kl"], false),
    ("x87", Unstable(sym::x87_target_feature), &[], false),
    ("xop", Unstable(sym::xop_target_feature), &[/*"fma4", */ "avx", "sse4a"], false),
    ("xsave", Stable, &[], false),
    ("xsavec", Stable, &["xsave"], false),
    ("xsaveopt", Stable, &["xsave"], false),
    ("xsaves", Stable, &["xsave"], false),
    // tidy-alphabetical-end
];

const HEXAGON_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("hvx", Unstable(sym::hexagon_target_feature), &[], false),
    ("hvx-length128b", Unstable(sym::hexagon_target_feature), &["hvx"], false),
    // tidy-alphabetical-end
];

static POWERPC_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("altivec", Unstable(sym::powerpc_target_feature), &[], false),
    ("msync", Unstable(sym::powerpc_target_feature), &[], false),
    ("partword-atomics", Unstable(sym::powerpc_target_feature), &[], false),
    ("power8-altivec", Unstable(sym::powerpc_target_feature), &["altivec"], false),
    ("power8-crypto", Unstable(sym::powerpc_target_feature), &["power8-altivec"], false),
    ("power8-vector", Unstable(sym::powerpc_target_feature), &["vsx", "power8-altivec"], false),
    ("power9-altivec", Unstable(sym::powerpc_target_feature), &["power8-altivec"], false),
    (
        "power9-vector",
        Unstable(sym::powerpc_target_feature),
        &["power8-vector", "power9-altivec"],
        false,
    ),
    ("power10-vector", Unstable(sym::powerpc_target_feature), &["power9-vector"], false),
    ("quadword-atomics", Unstable(sym::powerpc_target_feature), &[], false),
    ("vsx", Unstable(sym::powerpc_target_feature), &["altivec"], false),
    // tidy-alphabetical-end
];

const MIPS_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("fp64", Unstable(sym::mips_target_feature), &[], false),
    ("msa", Unstable(sym::mips_target_feature), &[], false),
    ("virt", Unstable(sym::mips_target_feature), &[], false),
    // tidy-alphabetical-end
];

const NVPTX_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("sm_20", Unstable(sym::nvptx_target_feature), &[], false),
    ("sm_21", Unstable(sym::nvptx_target_feature), &["sm_20"], false),
    ("sm_30", Unstable(sym::nvptx_target_feature), &["sm_21"], false),
    ("sm_32", Unstable(sym::nvptx_target_feature), &["sm_30"], false),
    ("sm_35", Unstable(sym::nvptx_target_feature), &["sm_32"], false),
    ("sm_37", Unstable(sym::nvptx_target_feature), &["sm_35"], false),
    ("sm_50", Unstable(sym::nvptx_target_feature), &["sm_37"], false),
    ("sm_52", Unstable(sym::nvptx_target_feature), &["sm_50"], false),
    ("sm_53", Unstable(sym::nvptx_target_feature), &["sm_52"], false),
    ("sm_60", Unstable(sym::nvptx_target_feature), &["sm_53"], false),
    ("sm_61", Unstable(sym::nvptx_target_feature), &["sm_60"], false),
    ("sm_62", Unstable(sym::nvptx_target_feature), &["sm_61"], false),
    ("sm_70", Unstable(sym::nvptx_target_feature), &["sm_62"], false),
    ("sm_72", Unstable(sym::nvptx_target_feature), &["sm_70"], false),
    ("sm_75", Unstable(sym::nvptx_target_feature), &["sm_72"], false),
    ("sm_80", Unstable(sym::nvptx_target_feature), &["sm_75"], false),
    ("sm_86", Unstable(sym::nvptx_target_feature), &["sm_80"], false),
    ("sm_87", Unstable(sym::nvptx_target_feature), &["sm_86"], false),
    ("sm_89", Unstable(sym::nvptx_target_feature), &["sm_87"], false),
    ("sm_90", Unstable(sym::nvptx_target_feature), &["sm_89"], false),
    ("sm_90a", Unstable(sym::nvptx_target_feature), &["sm_90"], false),
    // tidy-alphabetical-end
    // tidy-alphabetical-start
    ("sm_100", Unstable(sym::nvptx_target_feature), &["sm_90"], false),
    ("sm_100a", Unstable(sym::nvptx_target_feature), &["sm_100"], false),
    ("sm_101", Unstable(sym::nvptx_target_feature), &["sm_100"], false),
    ("sm_101a", Unstable(sym::nvptx_target_feature), &["sm_101"], false),
    ("sm_120", Unstable(sym::nvptx_target_feature), &["sm_101"], false),
    ("sm_120a", Unstable(sym::nvptx_target_feature), &["sm_120"], false),
    // tidy-alphabetical-end
    // tidy-alphabetical-start
    ("ptx32", Unstable(sym::nvptx_target_feature), &[], false),
    ("ptx40", Unstable(sym::nvptx_target_feature), &["ptx32"], false),
    ("ptx41", Unstable(sym::nvptx_target_feature), &["ptx40"], false),
    ("ptx42", Unstable(sym::nvptx_target_feature), &["ptx41"], false),
    ("ptx43", Unstable(sym::nvptx_target_feature), &["ptx42"], false),
    ("ptx50", Unstable(sym::nvptx_target_feature), &["ptx43"], false),
    ("ptx60", Unstable(sym::nvptx_target_feature), &["ptx50"], false),
    ("ptx61", Unstable(sym::nvptx_target_feature), &["ptx60"], false),
    ("ptx62", Unstable(sym::nvptx_target_feature), &["ptx61"], false),
    ("ptx63", Unstable(sym::nvptx_target_feature), &["ptx62"], false),
    ("ptx64", Unstable(sym::nvptx_target_feature), &["ptx63"], false),
    ("ptx65", Unstable(sym::nvptx_target_feature), &["ptx64"], false),
    ("ptx70", Unstable(sym::nvptx_target_feature), &["ptx65"], false),
    ("ptx71", Unstable(sym::nvptx_target_feature), &["ptx70"], false),
    ("ptx72", Unstable(sym::nvptx_target_feature), &["ptx71"], false),
    ("ptx73", Unstable(sym::nvptx_target_feature), &["ptx72"], false),
    ("ptx74", Unstable(sym::nvptx_target_feature), &["ptx73"], false),
    ("ptx75", Unstable(sym::nvptx_target_feature), &["ptx74"], false),
    ("ptx76", Unstable(sym::nvptx_target_feature), &["ptx75"], false),
    ("ptx77", Unstable(sym::nvptx_target_feature), &["ptx76"], false),
    ("ptx78", Unstable(sym::nvptx_target_feature), &["ptx77"], false),
    ("ptx80", Unstable(sym::nvptx_target_feature), &["ptx78"], false),
    ("ptx81", Unstable(sym::nvptx_target_feature), &["ptx80"], false),
    ("ptx82", Unstable(sym::nvptx_target_feature), &["ptx81"], false),
    ("ptx83", Unstable(sym::nvptx_target_feature), &["ptx82"], false),
    ("ptx84", Unstable(sym::nvptx_target_feature), &["ptx83"], false),
    ("ptx85", Unstable(sym::nvptx_target_feature), &["ptx84"], false),
    ("ptx86", Unstable(sym::nvptx_target_feature), &["ptx85"], false),
    ("ptx87", Unstable(sym::nvptx_target_feature), &["ptx86"], false),
    // tidy-alphabetical-end
];

static RISCV_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("a", Stable, &["zaamo", "zalrsc"], false),
    ("b", Unstable(sym::riscv_target_feature), &["zba", "zbb", "zbs"], false),
    ("c", Stable, &["zca"], false),
    ("d", Unstable(sym::riscv_target_feature), &["f"], false),
    ("e", Unstable(sym::riscv_target_feature), &[], false),
    ("f", Unstable(sym::riscv_target_feature), &["zicsr"], false),
    (
        "forced-atomics",
        Stability::Forbidden { reason: "unsound because it changes the ABI of atomic operations" },
        &[],
        false,
    ),
    ("m", Stable, &[], false),
    ("relax", Unstable(sym::riscv_target_feature), &[], false),
    (
        "rva23u64",
        Unstable(sym::riscv_target_feature),
        &[
            "m",
            "a",
            "f",
            "d",
            "c",
            "b",
            "v",
            "zicsr",
            "zicntr",
            "zihpm",
            "ziccif",
            "ziccrse",
            "ziccamoa",
            "zicclsm",
            "zic64b",
            "za64rs",
            "zihintpause",
            "zba",
            "zbb",
            "zbs",
            "zicbom",
            "zicbop",
            "zicboz",
            "zfhmin",
            "zkt",
            "zvfhmin",
            "zvbb",
            "zvkt",
            "zihintntl",
            "zicond",
            "zimop",
            "zcmop",
            "zcb",
            "zfa",
            "zawrs",
            "supm",
        ],
        false,
    ),
    ("supm", Unstable(sym::riscv_target_feature), &[], false),
    ("unaligned-scalar-mem", Unstable(sym::riscv_target_feature), &[], false),
    ("unaligned-vector-mem", Unstable(sym::riscv_target_feature), &[], false),
    ("v", Unstable(sym::riscv_target_feature), &["zvl128b", "zve64d"], false),
    ("za64rs", Unstable(sym::riscv_target_feature), &["za128rs"], false), // Za64rs ⊃ Za128rs
    ("za128rs", Unstable(sym::riscv_target_feature), &[], false),
    ("zaamo", Unstable(sym::riscv_target_feature), &[], false),
    ("zabha", Unstable(sym::riscv_target_feature), &["zaamo"], false),
    ("zacas", Unstable(sym::riscv_target_feature), &["zaamo"], false),
    ("zalrsc", Unstable(sym::riscv_target_feature), &[], false),
    ("zama16b", Unstable(sym::riscv_target_feature), &[], false),
    ("zawrs", Unstable(sym::riscv_target_feature), &[], false),
    ("zba", Stable, &[], false),
    ("zbb", Stable, &[], false),
    ("zbc", Stable, &["zbkc"], false), // Zbc ⊃ Zbkc
    ("zbkb", Stable, &[], false),
    ("zbkc", Stable, &[], false),
    ("zbkx", Stable, &[], false),
    ("zbs", Stable, &[], false),
    ("zca", Unstable(sym::riscv_target_feature), &[], false),
    ("zcb", Unstable(sym::riscv_target_feature), &["zca"], false),
    ("zcmop", Unstable(sym::riscv_target_feature), &["zca"], false),
    ("zdinx", Unstable(sym::riscv_target_feature), &["zfinx"], false),
    ("zfa", Unstable(sym::riscv_target_feature), &["f"], false),
    ("zfbfmin", Unstable(sym::riscv_target_feature), &["f"], false), // and a subset of Zfhmin
    ("zfh", Unstable(sym::riscv_target_feature), &["zfhmin"], false),
    ("zfhmin", Unstable(sym::riscv_target_feature), &["f"], false),
    ("zfinx", Unstable(sym::riscv_target_feature), &["zicsr"], false),
    ("zhinx", Unstable(sym::riscv_target_feature), &["zhinxmin"], false),
    ("zhinxmin", Unstable(sym::riscv_target_feature), &["zfinx"], false),
    ("zic64b", Unstable(sym::riscv_target_feature), &[], false),
    ("zicbom", Unstable(sym::riscv_target_feature), &[], false),
    ("zicbop", Unstable(sym::riscv_target_feature), &[], false),
    ("zicboz", Unstable(sym::riscv_target_feature), &[], false),
    ("ziccamoa", Unstable(sym::riscv_target_feature), &[], false),
    ("ziccif", Unstable(sym::riscv_target_feature), &[], false),
    ("zicclsm", Unstable(sym::riscv_target_feature), &[], false),
    ("ziccrse", Unstable(sym::riscv_target_feature), &[], false),
    ("zicntr", Unstable(sym::riscv_target_feature), &["zicsr"], false),
    ("zicond", Unstable(sym::riscv_target_feature), &[], false),
    ("zicsr", Unstable(sym::riscv_target_feature), &[], false),
    ("zifencei", Unstable(sym::riscv_target_feature), &[], false),
    ("zihintntl", Unstable(sym::riscv_target_feature), &[], false),
    ("zihintpause", Unstable(sym::riscv_target_feature), &[], false),
    ("zihpm", Unstable(sym::riscv_target_feature), &["zicsr"], false),
    ("zimop", Unstable(sym::riscv_target_feature), &[], false),
    ("zk", Stable, &["zkn", "zkr", "zkt"], false),
    ("zkn", Stable, &["zbkb", "zbkc", "zbkx", "zkne", "zknd", "zknh"], false),
    ("zknd", Stable, &[], false),
    ("zkne", Stable, &[], false),
    ("zknh", Stable, &[], false),
    ("zkr", Stable, &[], false),
    ("zks", Stable, &["zbkb", "zbkc", "zbkx", "zksed", "zksh"], false),
    ("zksed", Stable, &[], false),
    ("zksh", Stable, &[], false),
    ("zkt", Stable, &[], false),
    ("ztso", Unstable(sym::riscv_target_feature), &[], false),
    ("zvbb", Unstable(sym::riscv_target_feature), &["zvkb"], false), // Zvbb ⊃ Zvkb
    ("zvbc", Unstable(sym::riscv_target_feature), &["zve64x"], false),
    ("zve32f", Unstable(sym::riscv_target_feature), &["zve32x", "f"], false),
    ("zve32x", Unstable(sym::riscv_target_feature), &["zvl32b", "zicsr"], false),
    ("zve64d", Unstable(sym::riscv_target_feature), &["zve64f", "d"], false),
    ("zve64f", Unstable(sym::riscv_target_feature), &["zve32f", "zve64x"], false),
    ("zve64x", Unstable(sym::riscv_target_feature), &["zve32x", "zvl64b"], false),
    ("zvfbfmin", Unstable(sym::riscv_target_feature), &["zve32f"], false),
    ("zvfbfwma", Unstable(sym::riscv_target_feature), &["zfbfmin", "zvfbfmin"], false),
    ("zvfh", Unstable(sym::riscv_target_feature), &["zvfhmin", "zve32f", "zfhmin"], false), // Zvfh ⊃ Zvfhmin
    ("zvfhmin", Unstable(sym::riscv_target_feature), &["zve32f"], false),
    ("zvkb", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvkg", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvkn", Unstable(sym::riscv_target_feature), &["zvkned", "zvknhb", "zvkb", "zvkt"], false),
    ("zvknc", Unstable(sym::riscv_target_feature), &["zvkn", "zvbc"], false),
    ("zvkned", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvkng", Unstable(sym::riscv_target_feature), &["zvkn", "zvkg"], false),
    ("zvknha", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvknhb", Unstable(sym::riscv_target_feature), &["zvknha", "zve64x"], false), // Zvknhb ⊃ Zvknha
    ("zvks", Unstable(sym::riscv_target_feature), &["zvksed", "zvksh", "zvkb", "zvkt"], false),
    ("zvksc", Unstable(sym::riscv_target_feature), &["zvks", "zvbc"], false),
    ("zvksed", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvksg", Unstable(sym::riscv_target_feature), &["zvks", "zvkg"], false),
    ("zvksh", Unstable(sym::riscv_target_feature), &["zve32x"], false),
    ("zvkt", Unstable(sym::riscv_target_feature), &[], false),
    ("zvl32b", Unstable(sym::riscv_target_feature), &[], false),
    ("zvl64b", Unstable(sym::riscv_target_feature), &["zvl32b"], false),
    ("zvl128b", Unstable(sym::riscv_target_feature), &["zvl64b"], false),
    ("zvl256b", Unstable(sym::riscv_target_feature), &["zvl128b"], false),
    ("zvl512b", Unstable(sym::riscv_target_feature), &["zvl256b"], false),
    ("zvl1024b", Unstable(sym::riscv_target_feature), &["zvl512b"], false),
    ("zvl2048b", Unstable(sym::riscv_target_feature), &["zvl1024b"], false),
    ("zvl4096b", Unstable(sym::riscv_target_feature), &["zvl2048b"], false),
    ("zvl8192b", Unstable(sym::riscv_target_feature), &["zvl4096b"], false),
    ("zvl16384b", Unstable(sym::riscv_target_feature), &["zvl8192b"], false),
    ("zvl32768b", Unstable(sym::riscv_target_feature), &["zvl16384b"], false),
    ("zvl65536b", Unstable(sym::riscv_target_feature), &["zvl32768b"], false),
    // tidy-alphabetical-end
];

static WASM_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("atomics", Unstable(sym::wasm_target_feature), &[], true),
    ("bulk-memory", Stable, &[], false),
    ("exception-handling", Unstable(sym::wasm_target_feature), &[], true),
    ("extended-const", Stable, &[], false),
    ("multivalue", Stable, &[], false),
    ("mutable-globals", Stable, &[], false),
    ("nontrapping-fptoint", Stable, &[], false),
    ("reference-types", Stable, &[], false),
    ("relaxed-simd", Stable, &["simd128"], false),
    ("sign-ext", Stable, &[], false),
    ("simd128", Stable, &[], false),
    ("tail-call", Stable, &[], false),
    ("wide-arithmetic", Unstable(sym::wasm_target_feature), &[], false),
    // tidy-alphabetical-end
];

const BPF_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] =
    &[("alu32", Unstable(sym::bpf_target_feature), &[], false)];

static CSKY_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("2e3", Unstable(sym::csky_target_feature), &["e2"], false),
    ("3e3r1", Unstable(sym::csky_target_feature), &[], false),
    ("3e3r2", Unstable(sym::csky_target_feature), &["3e3r1", "doloop"], false),
    ("3e3r3", Unstable(sym::csky_target_feature), &["doloop"], false),
    ("3e7", Unstable(sym::csky_target_feature), &["2e3"], false),
    ("7e10", Unstable(sym::csky_target_feature), &["3e7"], false),
    ("10e60", Unstable(sym::csky_target_feature), &["7e10"], false),
    ("cache", Unstable(sym::csky_target_feature), &[], false),
    ("doloop", Unstable(sym::csky_target_feature), &[], false),
    ("dsp1e2", Unstable(sym::csky_target_feature), &[], false),
    ("dspe60", Unstable(sym::csky_target_feature), &[], false),
    ("e1", Unstable(sym::csky_target_feature), &["elrw"], false),
    ("e2", Unstable(sym::csky_target_feature), &["e2"], false),
    ("edsp", Unstable(sym::csky_target_feature), &[], false),
    ("elrw", Unstable(sym::csky_target_feature), &[], false),
    ("float1e2", Unstable(sym::csky_target_feature), &[], false),
    ("float1e3", Unstable(sym::csky_target_feature), &[], false),
    ("float3e4", Unstable(sym::csky_target_feature), &[], false),
    ("float7e60", Unstable(sym::csky_target_feature), &[], false),
    ("floate1", Unstable(sym::csky_target_feature), &[], false),
    ("hard-tp", Unstable(sym::csky_target_feature), &[], false),
    ("high-registers", Unstable(sym::csky_target_feature), &[], false),
    ("hwdiv", Unstable(sym::csky_target_feature), &[], false),
    ("mp", Unstable(sym::csky_target_feature), &["2e3"], false),
    ("mp1e2", Unstable(sym::csky_target_feature), &["3e7"], false),
    ("nvic", Unstable(sym::csky_target_feature), &[], false),
    ("trust", Unstable(sym::csky_target_feature), &[], false),
    ("vdsp2e60f", Unstable(sym::csky_target_feature), &[], false),
    ("vdspv1", Unstable(sym::csky_target_feature), &[], false),
    ("vdspv2", Unstable(sym::csky_target_feature), &[], false),
    // tidy-alphabetical-end
    //fpu
    // tidy-alphabetical-start
    ("fdivdu", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv2_df", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv2_sf", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv3_df", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv3_hf", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv3_hi", Unstable(sym::csky_target_feature), &[], false),
    ("fpuv3_sf", Unstable(sym::csky_target_feature), &[], false),
    ("hard-float", Unstable(sym::csky_target_feature), &[], false),
    ("hard-float-abi", Unstable(sym::csky_target_feature), &[], false),
    // tidy-alphabetical-end
];

static LOONGARCH_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("32s", Unstable(sym::loongarch_target_feature), &[], false),
    ("d", Stable, &["f"], false),
    ("div32", Unstable(sym::loongarch_target_feature), &[], false),
    ("f", Stable, &[], false),
    ("frecipe", Stable, &[], false),
    ("lam-bh", Unstable(sym::loongarch_target_feature), &[], false),
    ("lamcas", Unstable(sym::loongarch_target_feature), &[], false),
    ("lasx", Stable, &["lsx"], false),
    ("lbt", Stable, &[], false),
    ("ld-seq-sa", Unstable(sym::loongarch_target_feature), &[], false),
    ("lsx", Stable, &["d"], false),
    ("lvz", Stable, &[], false),
    ("relax", Unstable(sym::loongarch_target_feature), &[], false),
    ("scq", Unstable(sym::loongarch_target_feature), &[], false),
    ("ual", Unstable(sym::loongarch_target_feature), &[], false),
    // tidy-alphabetical-end
];

#[rustfmt::skip]
const IBMZ_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    // For "backchain", https://github.com/rust-lang/rust/issues/142412 is a stabilization blocker
    ("backchain", Unstable(sym::s390x_target_feature), &[], false),
    ("concurrent-functions", Unstable(sym::s390x_target_feature), &[], false),
    ("deflate-conversion", Unstable(sym::s390x_target_feature), &[], false),
    ("enhanced-sort", Unstable(sym::s390x_target_feature), &[], false),
    ("guarded-storage", Unstable(sym::s390x_target_feature), &[], false),
    ("high-word", Unstable(sym::s390x_target_feature), &[], false),
    // LLVM does not define message-security-assist-extension versions 1, 2, 6, 10 and 11.
    ("message-security-assist-extension3", Unstable(sym::s390x_target_feature), &[], false),
    ("message-security-assist-extension4", Unstable(sym::s390x_target_feature), &[], false),
    ("message-security-assist-extension5", Unstable(sym::s390x_target_feature), &[], false),
    ("message-security-assist-extension8", Unstable(sym::s390x_target_feature), &["message-security-assist-extension3"], false),
    ("message-security-assist-extension9", Unstable(sym::s390x_target_feature), &["message-security-assist-extension3", "message-security-assist-extension4"], false),
    ("message-security-assist-extension12", Unstable(sym::s390x_target_feature), &[], false),
    ("miscellaneous-extensions-2", Unstable(sym::s390x_target_feature), &[], false),
    ("miscellaneous-extensions-3", Unstable(sym::s390x_target_feature), &[], false),
    ("miscellaneous-extensions-4", Unstable(sym::s390x_target_feature), &[], false),
    ("nnp-assist", Unstable(sym::s390x_target_feature), &["vector"], false),
    ("soft-float", Forbidden { reason: "currently unsupported ABI-configuration feature" }, &[], false),
    ("transactional-execution", Unstable(sym::s390x_target_feature), &[], false),
    ("vector", Unstable(sym::s390x_target_feature), &[], false),
    ("vector-enhancements-1", Unstable(sym::s390x_target_feature), &["vector"], false),
    ("vector-enhancements-2", Unstable(sym::s390x_target_feature), &["vector-enhancements-1"], false),
    ("vector-enhancements-3", Unstable(sym::s390x_target_feature), &["vector-enhancements-2"], false),
    ("vector-packed-decimal", Unstable(sym::s390x_target_feature), &["vector"], false),
    ("vector-packed-decimal-enhancement", Unstable(sym::s390x_target_feature), &["vector-packed-decimal"], false),
    ("vector-packed-decimal-enhancement-2", Unstable(sym::s390x_target_feature), &["vector-packed-decimal-enhancement"], false),
    ("vector-packed-decimal-enhancement-3", Unstable(sym::s390x_target_feature), &["vector-packed-decimal-enhancement-2"], false),
    // tidy-alphabetical-end
];

const SPARC_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("leoncasa", Unstable(sym::sparc_target_feature), &[], false),
    ("v8plus", Unstable(sym::sparc_target_feature), &[], false),
    ("v9", Unstable(sym::sparc_target_feature), &[], false),
    // tidy-alphabetical-end
];

static M68K_FEATURES: &[(&str, Stability, ImpliedFeatures, TargetModifier)] = &[
    // tidy-alphabetical-start
    ("isa-68000", Unstable(sym::m68k_target_feature), &[], false),
    ("isa-68010", Unstable(sym::m68k_target_feature), &["isa-68000"], false),
    ("isa-68020", Unstable(sym::m68k_target_feature), &["isa-68010"], false),
    ("isa-68030", Unstable(sym::m68k_target_feature), &["isa-68020"], false),
    ("isa-68040", Unstable(sym::m68k_target_feature), &["isa-68030", "isa-68882"], false),
    ("isa-68060", Unstable(sym::m68k_target_feature), &["isa-68040"], false),
    // FPU
    ("isa-68881", Unstable(sym::m68k_target_feature), &[], false),
    ("isa-68882", Unstable(sym::m68k_target_feature), &["isa-68881"], false),
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
        .chain(NVPTX_FEATURES.iter())
        .chain(RISCV_FEATURES.iter())
        .chain(WASM_FEATURES.iter())
        .chain(BPF_FEATURES.iter())
        .chain(CSKY_FEATURES)
        .chain(LOONGARCH_FEATURES)
        .chain(IBMZ_FEATURES)
        .chain(SPARC_FEATURES)
        .chain(M68K_FEATURES)
        .cloned()
        .map(|(f, s, _, _)| (f, s))
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
    pub fn rust_target_features(
        &self,
    ) -> &'static [(&'static str, Stability, ImpliedFeatures, TargetModifier)] {
        match &*self.arch {
            "arm" => ARM_FEATURES,
            "aarch64" | "arm64ec" => AARCH64_FEATURES,
            "x86" | "x86_64" => X86_FEATURES,
            "hexagon" => HEXAGON_FEATURES,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_FEATURES,
            "nvptx64" => NVPTX_FEATURES,
            "powerpc" | "powerpc64" => POWERPC_FEATURES,
            "riscv32" | "riscv64" => RISCV_FEATURES,
            "wasm32" | "wasm64" => WASM_FEATURES,
            "bpf" => BPF_FEATURES,
            "csky" => CSKY_FEATURES,
            "loongarch32" | "loongarch64" => LOONGARCH_FEATURES,
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
            "loongarch32" | "loongarch64" => LOONGARCH_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "riscv32" | "riscv64" => RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "wasm32" | "wasm64" => WASM_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "s390x" => S390X_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "sparc" | "sparc64" => SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "hexagon" => HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI,
            "nvptx64" | "bpf" | "m68k" => &[], // no vector ABI
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
        let implied_features = self
            .rust_target_features()
            .iter()
            .map(|(f, _, i, _)| (f, i))
            .collect::<FxHashMap<_, _>>();

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
                        // LLVM will use float registers when `fp-armv8` is available, e.g. for
                        // calls to built-ins. The only way to ensure a consistent softfloat ABI
                        // on aarch64 is to never enable `fp-armv8`, so we enforce that.
                        // In Rust we tie `neon` and `fp-armv8` together, therefore `neon` is the
                        // feature we have to mark as incompatible.
                        FeatureConstraints { required: &[], incompatible: &["neon"] }
                    }
                    _ => {
                        // Everything else is assumed to use a hardfloat ABI. neon and fp-armv8 must be enabled.
                        // `FeatureConstraints` uses Rust feature names, hence only "neon" shows up.
                        FeatureConstraints { required: &["neon"], incompatible: &[] }
                    }
                }
            }
            "riscv32" | "riscv64" => {
                // RISC-V handles ABI in a very sane way, being fully explicit via `llvm_abiname`
                // about what the intended ABI is.
                match &*self.llvm_abiname {
                    "ilp32d" | "lp64d" => {
                        // Requires d (which implies f), incompatible with e and zfinx.
                        FeatureConstraints { required: &["d"], incompatible: &["e", "zfinx"] }
                    }
                    "ilp32f" | "lp64f" => {
                        // Requires f, incompatible with e and zfinx.
                        FeatureConstraints { required: &["f"], incompatible: &["e", "zfinx"] }
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
            "loongarch32" | "loongarch64" => {
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
            "s390x" => {
                // We don't currently support a softfloat target on this architecture.
                // As usual, we have to reject swapping the `soft-float` target feature.
                // The "vector" target feature does not affect the ABI for floats
                // because the vector and float registers overlap.
                FeatureConstraints { required: &[], incompatible: &["soft-float"] }
            }
            _ => NOTHING,
        }
    }
}
