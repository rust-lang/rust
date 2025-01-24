use std::iter::FromIterator;

#[cfg(feature = "master")]
use gccjit::Context;
use rustc_codegen_ssa::codegen_attrs::check_tied_features;
use rustc_codegen_ssa::errors::TargetFeatureDisableOrEnable;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::unord::UnordSet;
use rustc_session::Session;
use rustc_target::target_features::RUSTC_SPECIFIC_FEATURES;
use smallvec::{SmallVec, smallvec};

use crate::errors::{
    ForbiddenCTargetFeature, PossibleFeature, UnknownCTargetFeature, UnknownCTargetFeaturePrefix,
    UnstableCTargetFeature,
};

/// The list of GCC features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
pub(crate) fn global_gcc_features(sess: &Session, diagnostics: bool) -> Vec<String> {
    // Features that come earlier are overridden by conflicting features later in the string.
    // Typically we'll want more explicit settings to override the implicit ones, so:
    //
    // * Features from -Ctarget-cpu=*; are overridden by [^1]
    // * Features implied by --target; are overridden by
    // * Features from -Ctarget-feature; are overridden by
    // * function specific features.
    //
    // [^1]: target-cpu=native is handled here, other target-cpu values are handled implicitly
    // through GCC march implementation.
    //
    // FIXME(nagisa): it isn't clear what's the best interaction between features implied by
    // `-Ctarget-cpu` and `--target` are. On one hand, you'd expect CLI arguments to always
    // override anything that's implicit, so e.g. when there's no `--target` flag, features implied
    // the host target are overridden by `-Ctarget-cpu=*`. On the other hand, what about when both
    // `--target` and `-Ctarget-cpu=*` are specified? Both then imply some target features and both
    // flags are specified by the user on the CLI. It isn't as clear-cut which order of precedence
    // should be taken in cases like these.
    let mut features = vec![];

    // Features implied by an implicit or explicit `--target`.
    features.extend(sess.target.features.split(',').filter(|v| !v.is_empty()).map(String::from));

    // -Ctarget-features
    let known_features = sess.target.rust_target_features();
    let mut featsmap = FxHashMap::default();

    // Ensure that all ABI-required features are enabled, and the ABI-forbidden ones
    // are disabled.
    let abi_feature_constraints = sess.target.abi_required_features();
    let abi_incompatible_set =
        FxHashSet::from_iter(abi_feature_constraints.incompatible.iter().copied());

    // Compute implied features
    let mut all_rust_features = vec![];
    for feature in sess.opts.cg.target_feature.split(',') {
        if let Some(feature) = feature.strip_prefix('+') {
            all_rust_features.extend(
                UnordSet::from(sess.target.implied_target_features(std::iter::once(feature)))
                    .to_sorted_stable_ord()
                    .iter()
                    .map(|&&s| (true, s)),
            )
        } else if let Some(feature) = feature.strip_prefix('-') {
            // FIXME: Why do we not remove implied features on "-" here?
            // We do the equivalent above in `target_features_cfg`.
            // See <https://github.com/rust-lang/rust/issues/134792>.
            all_rust_features.push((false, feature));
        } else if !feature.is_empty() && diagnostics {
            sess.dcx().emit_warn(UnknownCTargetFeaturePrefix { feature });
        }
    }
    // Remove features that are meant for rustc, not codegen.
    all_rust_features.retain(|&(_, feature)| {
        // Retain if it is not a rustc feature
        !RUSTC_SPECIFIC_FEATURES.contains(&feature)
    });

    // Check feature validity.
    if diagnostics {
        for &(enable, feature) in &all_rust_features {
            let feature_state = known_features.iter().find(|&&(v, _, _)| v == feature);
            match feature_state {
                None => {
                    let rust_feature = known_features.iter().find_map(|&(rust_feature, _, _)| {
                        let gcc_features = to_gcc_features(sess, rust_feature);
                        if gcc_features.contains(&feature) && !gcc_features.contains(&rust_feature)
                        {
                            Some(rust_feature)
                        } else {
                            None
                        }
                    });
                    let unknown_feature = if let Some(rust_feature) = rust_feature {
                        UnknownCTargetFeature {
                            feature,
                            rust_feature: PossibleFeature::Some { rust_feature },
                        }
                    } else {
                        UnknownCTargetFeature { feature, rust_feature: PossibleFeature::None }
                    };
                    sess.dcx().emit_warn(unknown_feature);
                }
                Some(&(_, stability, _)) => {
                    if let Err(reason) = stability.toggle_allowed() {
                        sess.dcx().emit_warn(ForbiddenCTargetFeature {
                            feature,
                            enabled: if enable { "enabled" } else { "disabled" },
                            reason,
                        });
                    } else if stability.requires_nightly().is_some() {
                        // An unstable feature. Warn about using it. (It makes little sense
                        // to hard-error here since we just warn about fully unknown
                        // features above).
                        sess.dcx().emit_warn(UnstableCTargetFeature { feature });
                    }
                }
            }

            // Ensure that the features we enable/disable are compatible with the ABI.
            if enable {
                if abi_incompatible_set.contains(feature) {
                    sess.dcx().emit_warn(ForbiddenCTargetFeature {
                        feature,
                        enabled: "enabled",
                        reason: "this feature is incompatible with the target ABI",
                    });
                }
            } else {
                // FIXME: we have to request implied features here since
                // negative features do not handle implied features above.
                for &required in abi_feature_constraints.required.iter() {
                    let implied = sess.target.implied_target_features(std::iter::once(required));
                    if implied.contains(feature) {
                        sess.dcx().emit_warn(ForbiddenCTargetFeature {
                            feature,
                            enabled: "disabled",
                            reason: "this feature is required by the target ABI",
                        });
                    }
                }
            }

            // FIXME(nagisa): figure out how to not allocate a full hashset here.
            featsmap.insert(feature, enable);
        }
    }

    // To be sure the ABI-relevant features are all in the right state, we explicitly
    // (un)set them here. This means if the target spec sets those features wrong,
    // we will silently correct them rather than silently producing wrong code.
    // (The target sanity check tries to catch this, but we can't know which features are
    // enabled in GCC by default so we can't be fully sure about that check.)
    // We add these at the beginning of the list so that `-Ctarget-features` can
    // still override it... that's unsound, but more compatible with past behavior.
    all_rust_features.splice(
        0..0,
        abi_feature_constraints
            .required
            .iter()
            .map(|&f| (true, f))
            .chain(abi_feature_constraints.incompatible.iter().map(|&f| (false, f))),
    );

    // Translate this into GCC features.
    let feats =
        all_rust_features.iter().flat_map(|&(enable, feature)| {
            let enable_disable = if enable { '+' } else { '-' };
            // We run through `to_gcc_features` when
            // passing requests down to GCC. This means that all in-language
            // features also work on the command line instead of having two
            // different names when the GCC name and the Rust name differ.
            to_gcc_features(sess, feature)
                .iter()
                .flat_map(|feat| to_gcc_features(sess, feat).into_iter())
                .map(|feature| {
                    if enable_disable == '-' {
                        format!("-{}", feature)
                    } else {
                        feature.to_string()
                    }
                })
                .collect::<Vec<_>>()
        });
    features.extend(feats);

    if diagnostics {
        if let Some(f) = check_tied_features(sess, &featsmap) {
            sess.dcx().emit_err(TargetFeatureDisableOrEnable {
                features: f,
                span: None,
                missing_features: None,
            });
        }
    }

    features
}

// To find a list of GCC's names, check https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
pub fn to_gcc_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        // FIXME: seems like x87 does not exist?
        ("x86", "x87") => smallvec![],
        ("x86", "sse4.2") => smallvec!["sse4.2", "crc32"],
        ("x86", "pclmulqdq") => smallvec!["pclmul"],
        ("x86", "rdrand") => smallvec!["rdrnd"],
        ("x86", "bmi1") => smallvec!["bmi"],
        ("x86", "cmpxchg16b") => smallvec!["cx16"],
        ("x86", "avx512vaes") => smallvec!["vaes"],
        ("x86", "avx512gfni") => smallvec!["gfni"],
        ("x86", "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512vbmi2'.
        ("x86", "avx512vbmi2") => smallvec!["avx512vbmi2", "avx512bw"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512bitalg'.
        ("x86", "avx512bitalg") => smallvec!["avx512bitalg", "avx512bw"],
        ("aarch64", "rcpc2") => smallvec!["rcpc-immo"],
        ("aarch64", "dpb") => smallvec!["ccpp"],
        ("aarch64", "dpb2") => smallvec!["ccdp"],
        ("aarch64", "frintts") => smallvec!["fptoint"],
        ("aarch64", "fcma") => smallvec!["complxnum"],
        ("aarch64", "pmuv3") => smallvec!["perfmon"],
        ("aarch64", "paca") => smallvec!["pauth"],
        ("aarch64", "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In GCC neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        ("aarch64", "f32mm") => smallvec!["f32mm", "neon"],
        ("aarch64", "f64mm") => smallvec!["f64mm", "neon"],
        ("aarch64", "fhm") => smallvec!["fp16fml", "neon"],
        ("aarch64", "fp16") => smallvec!["fullfp16", "neon"],
        ("aarch64", "jsconv") => smallvec!["jsconv", "neon"],
        ("aarch64", "sve") => smallvec!["sve", "neon"],
        ("aarch64", "sve2") => smallvec!["sve2", "neon"],
        ("aarch64", "sve2-aes") => smallvec!["sve2-aes", "neon"],
        ("aarch64", "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        ("aarch64", "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        ("aarch64", "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
        (_, s) => smallvec![s],
    }
}

fn arch_to_gcc(name: &str) -> &str {
    match name {
        "M68020" => "68020",
        _ => name,
    }
}

fn handle_native(name: &str) -> &str {
    if name != "native" {
        return arch_to_gcc(name);
    }

    #[cfg(feature = "master")]
    {
        // Get the native arch.
        let context = Context::default();
        context.get_target_info().arch().unwrap().to_str().unwrap()
    }
    #[cfg(not(feature = "master"))]
    unimplemented!();
}

pub fn target_cpu(sess: &Session) -> &str {
    match sess.opts.cg.target_cpu {
        Some(ref name) => handle_native(name),
        None => handle_native(sess.target.cpu.as_ref()),
    }
}
