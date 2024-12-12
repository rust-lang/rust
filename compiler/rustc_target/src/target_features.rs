//! Declares Rust's target feature names for each target.
//! Note that these are similar to but not always identical to LLVM's feature names,
//! and Rust adds some features that do not correspond to LLVM features at all.
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_span::{Symbol, sym};

use crate::spec::Target;

/// Features that control behaviour of rustc, rather than the codegen.
/// These exist globally and are not in the target-specific lists below.
pub const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

/// Features that require special handling when passing to LLVM:
/// these are target-specific (i.e., must also be listed in the target-specific list below)
/// but do not correspond to an LLVM target feature.
pub const RUSTC_SPECIAL_FEATURES: &[&str] = &["backchain"];

/// Stability information for target features.
/// `Toggleability` is the type storing whether (un)stable features can be toggled:
/// this is initially a function since it can depend on `Target`, but for stable hashing
/// it needs to be something hashable to we have to make the type generic.
#[derive(Debug, Clone)]
pub enum Stability<Toggleability> {
    /// This target feature is stable, it can be used in `#[target_feature]` and
    /// `#[cfg(target_feature)]`.
    Stable {
        /// When enabling/disabling the feature via `-Ctarget-feature` or `#[target_feature]`,
        /// determine if that is allowed.
        allow_toggle: Toggleability,
    },
    /// This target feature is unstable. It is only present in `#[cfg(target_feature)]` on
    /// nightly and using it in `#[target_feature]` requires enabling the given nightly feature.
    Unstable {
        /// This must be a *language* feature, or else rustc will ICE when reporting a missing
        /// feature gate!
        nightly_feature: Symbol,
        /// See `Stable::allow_toggle` comment above.
        allow_toggle: Toggleability,
    },
    /// This feature can not be set via `-Ctarget-feature` or `#[target_feature]`, it can only be
    /// set in the target spec. It is never set in `cfg(target_feature)`. Used in
    /// particular for features that change the floating-point ABI.
    Forbidden { reason: &'static str },
}

/// Returns `Ok` if the toggle is allowed, `Err` with an explanation of not.
/// The `bool` indicates whether the feature is being enabled (`true`) or disabled.
pub type AllowToggleUncomputed = fn(&Target, bool) -> Result<(), &'static str>;

/// The computed result of whether a feature can be enabled/disabled on the current target.
#[derive(Debug, Clone)]
pub struct AllowToggleComputed {
    enable: Result<(), &'static str>,
    disable: Result<(), &'static str>,
}

/// `Stability` where `allow_toggle` has not been computed yet.
pub type StabilityUncomputed = Stability<AllowToggleUncomputed>;
/// `Stability` where `allow_toggle` has already been computed.
pub type StabilityComputed = Stability<AllowToggleComputed>;

impl<CTX, Toggleability: HashStable<CTX>> HashStable<CTX> for Stability<Toggleability> {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Stability::Stable { allow_toggle } => {
                allow_toggle.hash_stable(hcx, hasher);
            }
            Stability::Unstable { nightly_feature, allow_toggle } => {
                nightly_feature.hash_stable(hcx, hasher);
                allow_toggle.hash_stable(hcx, hasher);
            }
            Stability::Forbidden { reason } => {
                reason.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<CTX> HashStable<CTX> for AllowToggleComputed {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        let AllowToggleComputed { enable, disable } = self;
        enable.hash_stable(hcx, hasher);
        disable.hash_stable(hcx, hasher);
    }
}

impl<Toggleability> Stability<Toggleability> {
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
            Stability::Unstable { nightly_feature, .. } => Some(nightly_feature),
            Stability::Stable { .. } => None,
            Stability::Forbidden { .. } => panic!("forbidden features should not reach this far"),
        }
    }
}

impl StabilityUncomputed {
    pub fn compute_toggleability(&self, target: &Target) -> StabilityComputed {
        use Stability::*;
        let compute = |f: AllowToggleUncomputed| AllowToggleComputed {
            enable: f(target, true),
            disable: f(target, false),
        };
        match *self {
            Stable { allow_toggle } => Stable { allow_toggle: compute(allow_toggle) },
            Unstable { nightly_feature, allow_toggle } => {
                Unstable { nightly_feature, allow_toggle: compute(allow_toggle) }
            }
            Forbidden { reason } => Forbidden { reason },
        }
    }

    pub fn toggle_allowed(&self, target: &Target, enable: bool) -> Result<(), &'static str> {
        use Stability::*;
        match *self {
            Stable { allow_toggle } => allow_toggle(target, enable),
            Unstable { allow_toggle, .. } => allow_toggle(target, enable),
            Forbidden { reason } => Err(reason),
        }
    }
}

impl StabilityComputed {
    /// Returns whether the feature may be toggled via `#[target_feature]` or `-Ctarget-feature`.
    /// (It might still be nightly-only even if this returns `true`, so make sure to also check
    /// `requires_nightly`.)
    pub fn toggle_allowed(&self, enable: bool) -> Result<(), &'static str> {
        let allow_toggle = match self {
            Stability::Stable { allow_toggle } => allow_toggle,
            Stability::Unstable { allow_toggle, .. } => allow_toggle,
            Stability::Forbidden { reason } => return Err(reason),
        };
        if enable { allow_toggle.enable } else { allow_toggle.disable }
    }
}

// Constructors for the list below, defaulting to "always allow toggle".
const STABLE: StabilityUncomputed = Stability::Stable { allow_toggle: |_target, _enable| Ok(()) };
const fn unstable(nightly_feature: Symbol) -> StabilityUncomputed {
    Stability::Unstable { nightly_feature, allow_toggle: |_target, _enable| Ok(()) }
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

const ARM_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("aclass", unstable(sym::arm_target_feature), &[]),
    ("aes", unstable(sym::arm_target_feature), &["neon"]),
    ("crc", unstable(sym::arm_target_feature), &[]),
    ("d32", unstable(sym::arm_target_feature), &[]),
    ("dotprod", unstable(sym::arm_target_feature), &["neon"]),
    ("dsp", unstable(sym::arm_target_feature), &[]),
    ("fp-armv8", unstable(sym::arm_target_feature), &["vfp4"]),
    (
        "fpregs",
        Stability::Unstable {
            nightly_feature: sym::arm_target_feature,
            allow_toggle: |target: &Target, _enable| {
                // Only allow toggling this if the target has `soft-float` set. With `soft-float`,
                // `fpregs` isn't needed so changing it cannot affect the ABI.
                if target.has_feature("soft-float") {
                    Ok(())
                } else {
                    Err("unsound on hard-float targets because it changes float ABI")
                }
            },
        },
        &[],
    ),
    ("i8mm", unstable(sym::arm_target_feature), &["neon"]),
    ("mclass", unstable(sym::arm_target_feature), &[]),
    ("neon", unstable(sym::arm_target_feature), &["vfp3"]),
    ("rclass", unstable(sym::arm_target_feature), &[]),
    ("sha2", unstable(sym::arm_target_feature), &["neon"]),
    ("soft-float", Stability::Forbidden { reason: "unsound because it changes float ABI" }, &[]),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", unstable(sym::arm_target_feature), &[]),
    ("thumb2", unstable(sym::arm_target_feature), &[]),
    ("trustzone", unstable(sym::arm_target_feature), &[]),
    ("v5te", unstable(sym::arm_target_feature), &[]),
    ("v6", unstable(sym::arm_target_feature), &["v5te"]),
    ("v6k", unstable(sym::arm_target_feature), &["v6"]),
    ("v6t2", unstable(sym::arm_target_feature), &["v6k", "thumb2"]),
    ("v7", unstable(sym::arm_target_feature), &["v6t2"]),
    ("v8", unstable(sym::arm_target_feature), &["v7"]),
    ("vfp2", unstable(sym::arm_target_feature), &[]),
    ("vfp3", unstable(sym::arm_target_feature), &["vfp2", "d32"]),
    ("vfp4", unstable(sym::arm_target_feature), &["vfp3"]),
    ("virtualization", unstable(sym::arm_target_feature), &[]),
    // tidy-alphabetical-end
];

const AARCH64_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    // FEAT_AES & FEAT_PMULL
    ("aes", STABLE, &["neon"]),
    // FEAT_BF16
    ("bf16", STABLE, &[]),
    // FEAT_BTI
    ("bti", STABLE, &[]),
    // FEAT_CRC
    ("crc", STABLE, &[]),
    // FEAT_CSSC
    ("cssc", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_DIT
    ("dit", STABLE, &[]),
    // FEAT_DotProd
    ("dotprod", STABLE, &["neon"]),
    // FEAT_DPB
    ("dpb", STABLE, &[]),
    // FEAT_DPB2
    ("dpb2", STABLE, &["dpb"]),
    // FEAT_ECV
    ("ecv", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_F32MM
    ("f32mm", STABLE, &["sve"]),
    // FEAT_F64MM
    ("f64mm", STABLE, &["sve"]),
    // FEAT_FAMINMAX
    ("faminmax", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_FCMA
    ("fcma", STABLE, &["neon"]),
    // FEAT_FHM
    ("fhm", STABLE, &["fp16"]),
    // FEAT_FLAGM
    ("flagm", STABLE, &[]),
    // FEAT_FLAGM2
    ("flagm2", unstable(sym::aarch64_unstable_target_feature), &[]),
    ("fp-armv8", Stability::Forbidden { reason: "Rust ties `fp-armv8` to `neon`" }, &[]),
    // FEAT_FP16
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("fp16", STABLE, &["neon"]),
    // FEAT_FP8
    ("fp8", unstable(sym::aarch64_unstable_target_feature), &["faminmax", "lut", "bf16"]),
    // FEAT_FP8DOT2
    ("fp8dot2", unstable(sym::aarch64_unstable_target_feature), &["fp8dot4"]),
    // FEAT_FP8DOT4
    ("fp8dot4", unstable(sym::aarch64_unstable_target_feature), &["fp8fma"]),
    // FEAT_FP8FMA
    ("fp8fma", unstable(sym::aarch64_unstable_target_feature), &["fp8"]),
    // FEAT_FRINTTS
    ("frintts", STABLE, &[]),
    // FEAT_HBC
    ("hbc", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_I8MM
    ("i8mm", STABLE, &[]),
    // FEAT_JSCVT
    // Rust ties FP and Neon: https://github.com/rust-lang/rust/pull/91608
    ("jsconv", STABLE, &["neon"]),
    // FEAT_LOR
    ("lor", STABLE, &[]),
    // FEAT_LSE
    ("lse", STABLE, &[]),
    // FEAT_LSE128
    ("lse128", unstable(sym::aarch64_unstable_target_feature), &["lse"]),
    // FEAT_LSE2
    ("lse2", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_LUT
    ("lut", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_MOPS
    ("mops", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_MTE & FEAT_MTE2
    ("mte", STABLE, &[]),
    // FEAT_AdvSimd & FEAT_FP
    (
        "neon",
        Stability::Stable {
            allow_toggle: |target, enable| {
                if target.abi == "softfloat" {
                    // `neon` has no ABI implications for softfloat targets, we can allow this.
                    Ok(())
                } else if enable
                    && !target.has_neg_feature("fp-armv8")
                    && !target.has_neg_feature("neon")
                {
                    // neon is enabled by default, and has not been disabled, so enabling it again
                    // is redundant and we can permit it. Forbidding this would be a breaking change
                    // since this feature is stable.
                    Ok(())
                } else {
                    Err("unsound on hard-float targets because it changes float ABI")
                }
            },
        },
        &[],
    ),
    // FEAT_PAUTH (address authentication)
    ("paca", STABLE, &[]),
    // FEAT_PAUTH (generic authentication)
    ("pacg", STABLE, &[]),
    // FEAT_PAN
    ("pan", STABLE, &[]),
    // FEAT_PAuth_LR
    ("pauth-lr", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_PMUv3
    ("pmuv3", STABLE, &[]),
    // FEAT_RNG
    ("rand", STABLE, &[]),
    // FEAT_RAS & FEAT_RASv1p1
    ("ras", STABLE, &[]),
    // FEAT_LRCPC
    ("rcpc", STABLE, &[]),
    // FEAT_LRCPC2
    ("rcpc2", STABLE, &["rcpc"]),
    // FEAT_LRCPC3
    ("rcpc3", unstable(sym::aarch64_unstable_target_feature), &["rcpc2"]),
    // FEAT_RDM
    ("rdm", STABLE, &["neon"]),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled globally using -Zfixed-x18, not
    // #[target_feature].
    // Note that cfg(target_feature = "reserve-x18") is currently not set for
    // targets that reserve x18 by default.
    ("reserve-x18", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_SB
    ("sb", STABLE, &[]),
    // FEAT_SHA1 & FEAT_SHA256
    ("sha2", STABLE, &["neon"]),
    // FEAT_SHA512 & FEAT_SHA3
    ("sha3", STABLE, &["sha2"]),
    // FEAT_SM3 & FEAT_SM4
    ("sm4", STABLE, &["neon"]),
    // FEAT_SME
    ("sme", unstable(sym::aarch64_unstable_target_feature), &["bf16"]),
    // FEAT_SME_B16B16
    ("sme-b16b16", unstable(sym::aarch64_unstable_target_feature), &["bf16", "sme2", "sve-b16b16"]),
    // FEAT_SME_F16F16
    ("sme-f16f16", unstable(sym::aarch64_unstable_target_feature), &["sme2"]),
    // FEAT_SME_F64F64
    ("sme-f64f64", unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME_F8F16
    ("sme-f8f16", unstable(sym::aarch64_unstable_target_feature), &["sme-f8f32"]),
    // FEAT_SME_F8F32
    ("sme-f8f32", unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"]),
    // FEAT_SME_FA64
    ("sme-fa64", unstable(sym::aarch64_unstable_target_feature), &["sme", "sve2"]),
    // FEAT_SME_I16I64
    ("sme-i16i64", unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME_LUTv2
    ("sme-lutv2", unstable(sym::aarch64_unstable_target_feature), &[]),
    // FEAT_SME2
    ("sme2", unstable(sym::aarch64_unstable_target_feature), &["sme"]),
    // FEAT_SME2p1
    ("sme2p1", unstable(sym::aarch64_unstable_target_feature), &["sme2"]),
    // FEAT_SPE
    ("spe", STABLE, &[]),
    // FEAT_SSBS & FEAT_SSBS2
    ("ssbs", STABLE, &[]),
    // FEAT_SSVE_FP8FDOT2
    ("ssve-fp8dot2", unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8dot4"]),
    // FEAT_SSVE_FP8FDOT4
    ("ssve-fp8dot4", unstable(sym::aarch64_unstable_target_feature), &["ssve-fp8fma"]),
    // FEAT_SSVE_FP8FMA
    ("ssve-fp8fma", unstable(sym::aarch64_unstable_target_feature), &["sme2", "fp8"]),
    // FEAT_SVE
    // It was decided that SVE requires Neon: https://github.com/rust-lang/rust/pull/91608
    //
    // LLVM doesn't enable Neon for SVE. ARM indicates that they're separate, but probably always
    // exist together: https://developer.arm.com/documentation/102340/0100/New-features-in-SVE2
    //
    // "For backwards compatibility, Neon and VFP are required in the latest architectures."
    ("sve", STABLE, &["neon"]),
    // FEAT_SVE_B16B16 (SVE or SME Z-targeting instructions)
    ("sve-b16b16", unstable(sym::aarch64_unstable_target_feature), &["bf16"]),
    // FEAT_SVE2
    ("sve2", STABLE, &["sve"]),
    // FEAT_SVE_AES & FEAT_SVE_PMULL128
    ("sve2-aes", STABLE, &["sve2", "aes"]),
    // FEAT_SVE2_BitPerm
    ("sve2-bitperm", STABLE, &["sve2"]),
    // FEAT_SVE2_SHA3
    ("sve2-sha3", STABLE, &["sve2", "sha3"]),
    // FEAT_SVE2_SM4
    ("sve2-sm4", STABLE, &["sve2", "sm4"]),
    // FEAT_SVE2p1
    ("sve2p1", unstable(sym::aarch64_unstable_target_feature), &["sve2"]),
    // FEAT_TME
    ("tme", STABLE, &[]),
    ("v8.1a", unstable(sym::aarch64_ver_target_feature), &[
        "crc", "lse", "rdm", "pan", "lor", "vh",
    ]),
    ("v8.2a", unstable(sym::aarch64_ver_target_feature), &["v8.1a", "ras", "dpb"]),
    ("v8.3a", unstable(sym::aarch64_ver_target_feature), &[
        "v8.2a", "rcpc", "paca", "pacg", "jsconv",
    ]),
    ("v8.4a", unstable(sym::aarch64_ver_target_feature), &["v8.3a", "dotprod", "dit", "flagm"]),
    ("v8.5a", unstable(sym::aarch64_ver_target_feature), &["v8.4a", "ssbs", "sb", "dpb2", "bti"]),
    ("v8.6a", unstable(sym::aarch64_ver_target_feature), &["v8.5a", "bf16", "i8mm"]),
    ("v8.7a", unstable(sym::aarch64_ver_target_feature), &["v8.6a", "wfxt"]),
    ("v8.8a", unstable(sym::aarch64_ver_target_feature), &["v8.7a", "hbc", "mops"]),
    ("v8.9a", unstable(sym::aarch64_ver_target_feature), &["v8.8a", "cssc"]),
    ("v9.1a", unstable(sym::aarch64_ver_target_feature), &["v9a", "v8.6a"]),
    ("v9.2a", unstable(sym::aarch64_ver_target_feature), &["v9.1a", "v8.7a"]),
    ("v9.3a", unstable(sym::aarch64_ver_target_feature), &["v9.2a", "v8.8a"]),
    ("v9.4a", unstable(sym::aarch64_ver_target_feature), &["v9.3a", "v8.9a"]),
    ("v9.5a", unstable(sym::aarch64_ver_target_feature), &["v9.4a"]),
    ("v9a", unstable(sym::aarch64_ver_target_feature), &["v8.5a", "sve2"]),
    // FEAT_VHE
    ("vh", STABLE, &[]),
    // FEAT_WFxT
    ("wfxt", unstable(sym::aarch64_unstable_target_feature), &[]),
    // tidy-alphabetical-end
];

const AARCH64_TIED_FEATURES: &[&[&str]] = &[
    &["paca", "pacg"], // Together these represent `pauth` in LLVM
];

const X86_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("adx", STABLE, &[]),
    ("aes", STABLE, &["sse2"]),
    ("amx-bf16", unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-complex", unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-fp16", unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-int8", unstable(sym::x86_amx_intrinsics), &["amx-tile"]),
    ("amx-tile", unstable(sym::x86_amx_intrinsics), &[]),
    ("avx", STABLE, &["sse4.2"]),
    ("avx2", STABLE, &["avx"]),
    ("avx512bf16", unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512bitalg", unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512bw", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512cd", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512dq", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512f", unstable(sym::avx512_target_feature), &["avx2", "fma", "f16c"]),
    ("avx512fp16", unstable(sym::avx512_target_feature), &["avx512bw", "avx512vl", "avx512dq"]),
    ("avx512ifma", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vbmi", unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512vbmi2", unstable(sym::avx512_target_feature), &["avx512bw"]),
    ("avx512vl", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vnni", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vp2intersect", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avx512vpopcntdq", unstable(sym::avx512_target_feature), &["avx512f"]),
    ("avxifma", unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxneconvert", unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnni", unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnniint16", unstable(sym::avx512_target_feature), &["avx2"]),
    ("avxvnniint8", unstable(sym::avx512_target_feature), &["avx2"]),
    ("bmi1", STABLE, &[]),
    ("bmi2", STABLE, &[]),
    ("cmpxchg16b", STABLE, &[]),
    ("ermsb", unstable(sym::ermsb_target_feature), &[]),
    ("f16c", STABLE, &["avx"]),
    ("fma", STABLE, &["avx"]),
    ("fxsr", STABLE, &[]),
    ("gfni", unstable(sym::avx512_target_feature), &["sse2"]),
    ("lahfsahf", unstable(sym::lahfsahf_target_feature), &[]),
    ("lzcnt", STABLE, &[]),
    ("movbe", STABLE, &[]),
    ("pclmulqdq", STABLE, &["sse2"]),
    ("popcnt", STABLE, &[]),
    ("prfchw", unstable(sym::prfchw_target_feature), &[]),
    ("rdrand", STABLE, &[]),
    ("rdseed", STABLE, &[]),
    ("rtm", unstable(sym::rtm_target_feature), &[]),
    ("sha", STABLE, &["sse2"]),
    ("sha512", unstable(sym::sha512_sm_x86), &["avx2"]),
    ("sm3", unstable(sym::sha512_sm_x86), &["avx"]),
    ("sm4", unstable(sym::sha512_sm_x86), &["avx2"]),
    ("soft-float", Stability::Forbidden { reason: "unsound because it changes float ABI" }, &[]),
    ("sse", STABLE, &[]),
    ("sse2", STABLE, &["sse"]),
    ("sse3", STABLE, &["sse2"]),
    ("sse4.1", STABLE, &["ssse3"]),
    ("sse4.2", STABLE, &["sse4.1"]),
    ("sse4a", unstable(sym::sse4a_target_feature), &["sse3"]),
    ("ssse3", STABLE, &["sse3"]),
    ("tbm", unstable(sym::tbm_target_feature), &[]),
    ("vaes", unstable(sym::avx512_target_feature), &["avx2", "aes"]),
    ("vpclmulqdq", unstable(sym::avx512_target_feature), &["avx", "pclmulqdq"]),
    (
        "x87",
        Stability::Unstable {
            nightly_feature: sym::x87_target_feature,
            allow_toggle: |target: &Target, _enable| {
                // Only allow toggling this if the target has `soft-float` set. With `soft-float`,
                // `fpregs` isn't needed so changing it cannot affect the ABI.
                if target.has_feature("soft-float") {
                    Ok(())
                } else {
                    Err("unsound on hard-float targets because it changes float ABI")
                }
            },
        },
        &[],
    ),
    ("xop", unstable(sym::xop_target_feature), &[/*"fma4", */ "avx", "sse4a"]),
    ("xsave", STABLE, &[]),
    ("xsavec", STABLE, &["xsave"]),
    ("xsaveopt", STABLE, &["xsave"]),
    ("xsaves", STABLE, &["xsave"]),
    // tidy-alphabetical-end
];

const HEXAGON_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("hvx", unstable(sym::hexagon_target_feature), &[]),
    ("hvx-length128b", unstable(sym::hexagon_target_feature), &["hvx"]),
    // tidy-alphabetical-end
];

const POWERPC_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("altivec", unstable(sym::powerpc_target_feature), &[]),
    ("partword-atomics", unstable(sym::powerpc_target_feature), &[]),
    ("power10-vector", unstable(sym::powerpc_target_feature), &["power9-vector"]),
    ("power8-altivec", unstable(sym::powerpc_target_feature), &["altivec"]),
    ("power8-crypto", unstable(sym::powerpc_target_feature), &["power8-altivec"]),
    ("power8-vector", unstable(sym::powerpc_target_feature), &["vsx", "power8-altivec"]),
    ("power9-altivec", unstable(sym::powerpc_target_feature), &["power8-altivec"]),
    ("power9-vector", unstable(sym::powerpc_target_feature), &["power8-vector", "power9-altivec"]),
    ("quadword-atomics", unstable(sym::powerpc_target_feature), &[]),
    ("vsx", unstable(sym::powerpc_target_feature), &["altivec"]),
    // tidy-alphabetical-end
];

const MIPS_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("fp64", unstable(sym::mips_target_feature), &[]),
    ("msa", unstable(sym::mips_target_feature), &[]),
    ("virt", unstable(sym::mips_target_feature), &[]),
    // tidy-alphabetical-end
];

const RISCV_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("a", STABLE, &["zaamo", "zalrsc"]),
    ("c", STABLE, &[]),
    (
        "d",
        Stability::Unstable {
            nightly_feature: sym::riscv_target_feature,
            allow_toggle: |target, enable| match &*target.llvm_abiname {
                "ilp32d" | "lp64d" if !enable => {
                    // The ABI requires the `d` feature, so it cannot be disabled.
                    Err("feature is required by ABI")
                }
                "ilp32e" if enable => {
                    // ilp32e is incompatible with features that need aligned load/stores > 32 bits,
                    // like `d`.
                    Err("feature is incompatible with ABI")
                }
                _ => Ok(()),
            },
        },
        &["f"],
    ),
    (
        "e",
        Stability::Unstable {
            // Given that this is a negative feature, consider this before stabilizing:
            // does it really make sense to enable this feature in an individual
            // function with `#[target_feature]`?
            nightly_feature: sym::riscv_target_feature,
            allow_toggle: |target, enable| {
                match &*target.llvm_abiname {
                    _ if !enable => {
                        // Disabling this feature means we can use more registers (x16-x31).
                        // The "e" ABIs treat them as caller-save, so it is safe to use them only
                        // in some parts of a program while the rest doesn't know they even exist.
                        // On other ABIs, the feature is already disabled anyway.
                        Ok(())
                    }
                    "ilp32e" | "lp64e" => {
                        // Embedded ABIs should already have the feature anyway, it's fine to enable
                        // it again from an ABI perspective.
                        Ok(())
                    }
                    _ => {
                        // *Not* an embedded ABI. Enabling `e` is invalid.
                        Err("feature is incompatible with ABI")
                    }
                }
            },
        },
        &[],
    ),
    (
        "f",
        Stability::Unstable {
            nightly_feature: sym::riscv_target_feature,
            allow_toggle: |target, enable| {
                match &*target.llvm_abiname {
                    "ilp32f" | "ilp32d" | "lp64f" | "lp64d" if !enable => {
                        // The ABI requires the `f` feature, so it cannot be disabled.
                        Err("feature is required by ABI")
                    }
                    _ => Ok(()),
                }
            },
        },
        &[],
    ),
    (
        "forced-atomics",
        Stability::Forbidden { reason: "unsound because it changes the ABI of atomic operations" },
        &[],
    ),
    ("m", STABLE, &[]),
    ("relax", unstable(sym::riscv_target_feature), &[]),
    ("unaligned-scalar-mem", unstable(sym::riscv_target_feature), &[]),
    ("v", unstable(sym::riscv_target_feature), &[]),
    ("zaamo", unstable(sym::riscv_target_feature), &[]),
    ("zabha", unstable(sym::riscv_target_feature), &["zaamo"]),
    ("zalrsc", unstable(sym::riscv_target_feature), &[]),
    ("zba", STABLE, &[]),
    ("zbb", STABLE, &[]),
    ("zbc", STABLE, &[]),
    ("zbkb", STABLE, &[]),
    ("zbkc", STABLE, &[]),
    ("zbkx", STABLE, &[]),
    ("zbs", STABLE, &[]),
    ("zdinx", unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zfh", unstable(sym::riscv_target_feature), &["zfhmin"]),
    ("zfhmin", unstable(sym::riscv_target_feature), &["f"]),
    ("zfinx", unstable(sym::riscv_target_feature), &[]),
    ("zhinx", unstable(sym::riscv_target_feature), &["zhinxmin"]),
    ("zhinxmin", unstable(sym::riscv_target_feature), &["zfinx"]),
    ("zk", STABLE, &["zkn", "zkr", "zkt"]),
    ("zkn", STABLE, &["zbkb", "zbkc", "zbkx", "zkne", "zknd", "zknh"]),
    ("zknd", STABLE, &[]),
    ("zkne", STABLE, &[]),
    ("zknh", STABLE, &[]),
    ("zkr", STABLE, &[]),
    ("zks", STABLE, &["zbkb", "zbkc", "zbkx", "zksed", "zksh"]),
    ("zksed", STABLE, &[]),
    ("zksh", STABLE, &[]),
    ("zkt", STABLE, &[]),
    // tidy-alphabetical-end
];

const WASM_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("atomics", unstable(sym::wasm_target_feature), &[]),
    ("bulk-memory", STABLE, &[]),
    ("exception-handling", unstable(sym::wasm_target_feature), &[]),
    ("extended-const", STABLE, &[]),
    ("multivalue", STABLE, &[]),
    ("mutable-globals", STABLE, &[]),
    ("nontrapping-fptoint", STABLE, &[]),
    ("reference-types", STABLE, &[]),
    ("relaxed-simd", STABLE, &["simd128"]),
    ("sign-ext", STABLE, &[]),
    ("simd128", STABLE, &[]),
    ("tail-call", STABLE, &[]),
    ("wide-arithmetic", unstable(sym::wasm_target_feature), &[]),
    // tidy-alphabetical-end
];

const BPF_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] =
    &[("alu32", unstable(sym::bpf_target_feature), &[])];

const CSKY_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("10e60", unstable(sym::csky_target_feature), &["7e10"]),
    ("2e3", unstable(sym::csky_target_feature), &["e2"]),
    ("3e3r1", unstable(sym::csky_target_feature), &[]),
    ("3e3r2", unstable(sym::csky_target_feature), &["3e3r1", "doloop"]),
    ("3e3r3", unstable(sym::csky_target_feature), &["doloop"]),
    ("3e7", unstable(sym::csky_target_feature), &["2e3"]),
    ("7e10", unstable(sym::csky_target_feature), &["3e7"]),
    ("cache", unstable(sym::csky_target_feature), &[]),
    ("doloop", unstable(sym::csky_target_feature), &[]),
    ("dsp1e2", unstable(sym::csky_target_feature), &[]),
    ("dspe60", unstable(sym::csky_target_feature), &[]),
    ("e1", unstable(sym::csky_target_feature), &["elrw"]),
    ("e2", unstable(sym::csky_target_feature), &["e2"]),
    ("edsp", unstable(sym::csky_target_feature), &[]),
    ("elrw", unstable(sym::csky_target_feature), &[]),
    ("float1e2", unstable(sym::csky_target_feature), &[]),
    ("float1e3", unstable(sym::csky_target_feature), &[]),
    ("float3e4", unstable(sym::csky_target_feature), &[]),
    ("float7e60", unstable(sym::csky_target_feature), &[]),
    ("floate1", unstable(sym::csky_target_feature), &[]),
    ("hard-tp", unstable(sym::csky_target_feature), &[]),
    ("high-registers", unstable(sym::csky_target_feature), &[]),
    ("hwdiv", unstable(sym::csky_target_feature), &[]),
    ("mp", unstable(sym::csky_target_feature), &["2e3"]),
    ("mp1e2", unstable(sym::csky_target_feature), &["3e7"]),
    ("nvic", unstable(sym::csky_target_feature), &[]),
    ("trust", unstable(sym::csky_target_feature), &[]),
    ("vdsp2e60f", unstable(sym::csky_target_feature), &[]),
    ("vdspv1", unstable(sym::csky_target_feature), &[]),
    ("vdspv2", unstable(sym::csky_target_feature), &[]),
    // tidy-alphabetical-end
    //fpu
    // tidy-alphabetical-start
    ("fdivdu", unstable(sym::csky_target_feature), &[]),
    ("fpuv2_df", unstable(sym::csky_target_feature), &[]),
    ("fpuv2_sf", unstable(sym::csky_target_feature), &[]),
    ("fpuv3_df", unstable(sym::csky_target_feature), &[]),
    ("fpuv3_hf", unstable(sym::csky_target_feature), &[]),
    ("fpuv3_hi", unstable(sym::csky_target_feature), &[]),
    ("fpuv3_sf", unstable(sym::csky_target_feature), &[]),
    ("hard-float", unstable(sym::csky_target_feature), &[]),
    ("hard-float-abi", unstable(sym::csky_target_feature), &[]),
    // tidy-alphabetical-end
];

const LOONGARCH_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("d", unstable(sym::loongarch_target_feature), &["f"]),
    ("f", unstable(sym::loongarch_target_feature), &[]),
    ("frecipe", unstable(sym::loongarch_target_feature), &[]),
    ("lasx", unstable(sym::loongarch_target_feature), &["lsx"]),
    ("lbt", unstable(sym::loongarch_target_feature), &[]),
    ("lsx", unstable(sym::loongarch_target_feature), &["d"]),
    ("lvz", unstable(sym::loongarch_target_feature), &[]),
    ("relax", unstable(sym::loongarch_target_feature), &[]),
    ("ual", unstable(sym::loongarch_target_feature), &[]),
    // tidy-alphabetical-end
];

const IBMZ_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("backchain", unstable(sym::s390x_target_feature), &[]),
    ("vector", unstable(sym::s390x_target_feature), &[]),
    // tidy-alphabetical-end
];

const SPARC_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("leoncasa", unstable(sym::sparc_target_feature), &[]),
    ("v8plus", unstable(sym::sparc_target_feature), &[]),
    ("v9", unstable(sym::sparc_target_feature), &[]),
    // tidy-alphabetical-end
];

const M68K_FEATURES: &[(&str, StabilityUncomputed, ImpliedFeatures)] = &[
    // tidy-alphabetical-start
    ("isa-68000", unstable(sym::m68k_target_feature), &[]),
    ("isa-68010", unstable(sym::m68k_target_feature), &["isa-68000"]),
    ("isa-68020", unstable(sym::m68k_target_feature), &["isa-68010"]),
    ("isa-68030", unstable(sym::m68k_target_feature), &["isa-68020"]),
    ("isa-68040", unstable(sym::m68k_target_feature), &["isa-68030", "isa-68882"]),
    ("isa-68060", unstable(sym::m68k_target_feature), &["isa-68040"]),
    // FPU
    ("isa-68881", unstable(sym::m68k_target_feature), &[]),
    ("isa-68882", unstable(sym::m68k_target_feature), &["isa-68881"]),
    // tidy-alphabetical-end
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primitives may be documented.
///
/// IMPORTANT: If you're adding another feature list above, make sure to add it to this iterator!
pub fn all_rust_features() -> impl Iterator<Item = (&'static str, StabilityUncomputed)> {
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
const RISCV_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[/*(64, "zvl64b"), */ (128, "v")];
// Always warn on SPARC, as the necessary target features cannot be enabled in Rust at the moment.
const SPARC_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[/*(64, "vis")*/];

const HEXAGON_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[/*(512, "hvx-length64b"),*/ (1024, "hvx-length128b")];
const MIPS_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "msa")];
const CSKY_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] = &[(128, "vdspv1")];
const LOONGARCH_FEATURES_FOR_CORRECT_VECTOR_ABI: &'static [(u64, &'static str)] =
    &[(128, "lsx"), (256, "lasx")];

impl Target {
    pub fn rust_target_features(
        &self,
    ) -> &'static [(&'static str, StabilityUncomputed, ImpliedFeatures)] {
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
