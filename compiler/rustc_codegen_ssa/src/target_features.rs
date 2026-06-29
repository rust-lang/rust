use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir::attrs::InstructionSetAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_middle::middle::codegen_fn_attrs::{TargetFeature, TargetFeatureKind};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::diagnostics::feature_err;
use rustc_session::lint::builtin::AARCH64_SOFTFLOAT_NEON;
use rustc_span::{Span, Symbol, edit_distance, sym};
use rustc_target::spec::{Arch, SanitizerSet};
use rustc_target::target_features::{RUSTC_SPECIFIC_FEATURES, Stability};
use smallvec::SmallVec;

use crate::diagnostics::{CrossArchFeatureNote, FeatureNotValid, FeatureNotValidHint};
use crate::{diagnostics, target_features};

/// Compute the enabled target features from the `#[target_feature]` function attribute.
/// Enabled target features are added to `target_features`.
pub(crate) fn from_target_feature_attr(
    tcx: TyCtxt<'_>,
    did: LocalDefId,
    features: &[(Symbol, Span)],
    was_forced: bool,
    rust_target_features: &UnordMap<String, target_features::Stability>,
    target_features: &mut Vec<TargetFeature>,
) {
    let rust_features = tcx.features();
    let abi_feature_constraints = tcx.sess.target.abi_required_features();
    for &(feature, feature_span) in features {
        let feature_str = feature.as_str();
        let Some(stability) = rust_target_features.get(feature_str) else {
            let hint = if let Some(stripped) = feature_str.strip_prefix('+')
                && rust_target_features.contains_key(stripped)
            {
                FeatureNotValidHint::RemovePlusFromFeatureName { span: feature_span, stripped }
            } else {
                // Show the 5 feature names that are most similar to the input.
                let mut valid_names: Vec<_> =
                    rust_target_features.keys().map(|name| name.as_str()).into_sorted_stable_ord();
                valid_names.sort_by_key(|name| {
                    edit_distance::edit_distance(name, feature.as_str(), 5).unwrap_or(usize::MAX)
                });
                valid_names.truncate(5);

                FeatureNotValidHint::ValidFeatureNames {
                    possibilities: valid_names.into(),
                    and_more: rust_target_features.len().saturating_sub(5),
                }
            };
            tcx.dcx().emit_err(FeatureNotValid {
                feature: feature_str,
                span: feature_span,
                hint,
                cross_arch: {
                    let arches = rustc_target::target_features::feature_to_arch_names(feature_str);
                    match arches {
                        [] => None,
                        [arch] => Some(CrossArchFeatureNote::Single { feature: feature_str, arch }),
                        [..] => Some(CrossArchFeatureNote::Multiple {
                            feature: feature_str,
                            arches: arches.into(),
                        }),
                    }
                },
            });
            continue;
        };

        // Only allow target features whose feature gates have been enabled
        // and which are permitted to be toggled.
        if let Err(reason) = stability.toggle_allowed() {
            tcx.dcx().emit_err(diagnostics::InternalOnlyTargetFeatureAttr {
                span: feature_span,
                feature: feature_str,
                reason,
            });
        } else if let Some(nightly_feature) = stability.requires_nightly(/* in_cfg */ false)
            && !rust_features.enabled(nightly_feature)
        {
            let explain = if stability.is_cfg_stable_toggle_unstable() {
                format!("the target feature `{feature}` is allowed in cfg but unstable otherwise")
            } else {
                format!("the target feature `{feature}` is currently unstable")
            };
            feature_err(&tcx.sess, nightly_feature, feature_span, explain).emit();
        } else {
            // Add this and the implied features.
            for &name in tcx.implied_target_features(feature) {
                // But ensure the ABI does not forbid enabling this.
                // Here we do assume that the backend doesn't add even more implied features
                // we don't know about, at least no features that would have ABI effects!
                // We skip this logic in rustdoc, where we want to allow all target features of
                // all targets, so we can't check their ABI compatibility and anyway we are not
                // generating code so "it's fine".
                if !tcx.sess.opts.actually_rustdoc {
                    if abi_feature_constraints.incompatible.contains(&name.as_str()) {
                        // For "neon" specifically, we emit an FCW instead of a hard error.
                        // See <https://github.com/rust-lang/rust/issues/134375>.
                        if tcx.sess.target.arch == Arch::AArch64 && name.as_str() == "neon" {
                            tcx.emit_node_span_lint(
                                AARCH64_SOFTFLOAT_NEON,
                                tcx.local_def_id_to_hir_id(did),
                                feature_span,
                                diagnostics::Aarch64SoftfloatNeon,
                            );
                        } else {
                            tcx.dcx().emit_err(diagnostics::InternalOnlyTargetFeatureAttr {
                                span: feature_span,
                                feature: name.as_str(),
                                reason: "this feature is incompatible with the target ABI",
                            });
                        }
                    }
                }
                let kind = if name != feature {
                    TargetFeatureKind::Implied
                } else if was_forced {
                    TargetFeatureKind::Forced
                } else {
                    TargetFeatureKind::Enabled
                };
                target_features.push(TargetFeature { name, kind });

                if !rust_target_features
                    .get(name.as_str())
                    .is_some_and(|s| s.toggle_allowed().is_ok())
                {
                    tcx.dcx().span_delayed_bug(
                        feature_span,
                        format!("internal-only feature {name} should not be toggled by `#[target_feature]`"),
                    );
                }
            }
        }
    }
}

/// Computes the set of target features used in a function for the purposes of
/// inline assembly.
fn asm_target_features(tcx: TyCtxt<'_>, did: DefId) -> &FxIndexSet<Symbol> {
    let mut target_features = tcx.sess.internal_target_features.clone();
    if tcx.def_kind(did).has_codegen_attrs() {
        let attrs = tcx.codegen_fn_attrs(did);
        target_features.extend(attrs.target_features.iter().map(|feature| feature.name));
        match attrs.instruction_set {
            None => {}
            Some(InstructionSetAttr::ArmA32) => {
                // FIXME(#120456) - is `swap_remove` correct?
                target_features.swap_remove(&sym::thumb_mode);
            }
            Some(InstructionSetAttr::ArmT32) => {
                target_features.insert(sym::thumb_mode);
            }
        }
    }

    tcx.arena.alloc(target_features)
}

/// Checks the function annotated with `#[target_feature]` is not a safe
/// trait method implementation, reporting an error if it is.
pub(crate) fn check_target_feature_trait_unsafe(tcx: TyCtxt<'_>, id: LocalDefId, attr_span: Span) {
    if let DefKind::AssocFn = tcx.def_kind(id) {
        let parent_id = tcx.local_parent(id);
        if let DefKind::Trait | DefKind::Impl { of_trait: true } = tcx.def_kind(parent_id) {
            tcx.dcx().emit_err(diagnostics::TargetFeatureSafeTrait {
                span: attr_span,
                def: tcx.def_span(id),
            });
        }
    }
}

/// Parse the value of the target spec `features` field or `-Ctarget-feature`, calling the closure
/// for each entry in the list, also expanding implied features (but only for actual Rust target
/// features). If the list contains a syntactically invalid item (not starting with `+`/`-`) , the
/// error callback is invoked.
fn parse_rust_feature_list<'a>(
    sess: &'a Session,
    features: &'a str,
    err_callback: impl Fn(&'a str),
    mut callback: impl FnMut(
        /* base_feature */ &'a str,
        /* with_implied */ Option<FxHashSet<&'a str>>,
        /* enable */ bool,
    ),
) {
    // A cache for the forward and backwards feature maps.
    let mut features_map: Option<FxHashMap<&str, _>> = None;
    let mut inverse_implied_features: Option<FxHashMap<&str, FxHashSet<&str>>> = None;

    for feature in features.split(',') {
        if let Some(base_feature) = feature.strip_prefix('+') {
            // Skip features that are not target features, but rustc features.
            if RUSTC_SPECIFIC_FEATURES.contains(&base_feature) {
                continue;
            }

            let features_map =
                features_map.get_or_insert_with(|| sess.target.rust_target_features_map());

            if !features_map.contains_key(&base_feature) {
                callback(base_feature, None, true);
                continue;
            }

            let implied_features = sess.target.implied_target_features(base_feature, &features_map);
            callback(base_feature, Some(implied_features), true)
        } else if let Some(base_feature) = feature.strip_prefix('-') {
            // Skip features that are not target features, but rustc features.
            if RUSTC_SPECIFIC_FEATURES.contains(&base_feature) {
                continue;
            }

            let features_map =
                features_map.get_or_insert_with(|| sess.target.rust_target_features_map());

            if !features_map.contains_key(&base_feature) {
                callback(base_feature, None, false);
                continue;
            }

            // If `f1` implies `f2`, then `!f2` implies `!f1` -- this is standard logical
            // contraposition. So we have to find all the reverse implications of `base_feature` and
            // disable them, too.

            let inverse_implied_features = inverse_implied_features.get_or_insert_with(|| {
                let mut set: FxHashMap<&str, FxHashSet<&str>> = FxHashMap::default();
                for (f, _, is) in sess.target.rust_target_features() {
                    for i in is.iter() {
                        set.entry(i).or_default().insert(f);
                    }
                }
                set
            });

            // Inverse implied target features have their own inverse implied target features, so we
            // traverse the map until there are no more features to add.
            let mut implied_features = FxHashSet::default();
            let mut new_features = vec![base_feature];
            while let Some(new_feature) = new_features.pop() {
                if implied_features.insert(new_feature) {
                    if let Some(implied_features) = inverse_implied_features.get(&new_feature) {
                        #[allow(rustc::potential_query_instability)]
                        new_features.extend(implied_features)
                    }
                }
            }

            callback(base_feature, Some(implied_features), false)
        } else if !feature.is_empty() {
            err_callback(feature)
        }
    }
}

/// Utility function for a codegen backend to compute the set of all actually enabled Rust target
/// features (which will be stored in `sess.internal_target_features`).
///
/// `to_backend_features` converts a Rust feature name into a list of backend feature names; this is
/// used for diagnostic purposes only.
///
/// `target_base_has_feature` should check whether the given feature (a Rust feature name!) is
/// enabled in the "base" target machine, i.e., without applying `-Ctarget-feature`. Note that LLVM
/// may consider features to be implied that we do not and vice-versa. We want `cfg` to be entirely
/// consistent with Rust feature implications, and thus only consult LLVM to expand the target CPU
/// to target features.
///
/// We do not have to worry about RUSTC_SPECIFIC_FEATURES here, those are handled elsewhere.
pub fn internal_target_features<'a, const N: usize>(
    sess: &Session,
    to_backend_features: impl Fn(&'a str) -> SmallVec<[&'a str; N]>,
    mut target_base_has_feature: impl FnMut(&str) -> bool,
) -> UnordSet<Symbol> {
    let features_map = sess.target.rust_target_features_map();

    // Compute which of the known target features are enabled in the 'base' target machine: for
    // every Rust target feature, ask the backend if it is enabled.
    let mut features: UnordSet<Symbol> = sess
        .target
        .rust_target_features()
        .iter()
        .filter(|(feature, _, _)| target_base_has_feature(feature))
        .flat_map(|(base_feature, _, _)| {
            // Expand the direct base feature into all transitively-implied features. Note that we
            // cannot simply use the `implied` field of the tuple since that only contains
            // directly-implied features.
            //
            // Iteration order is irrelevant because we're collecting into an `UnordSet`.
            #[allow(rustc::potential_query_instability)]
            sess.target
                .implied_target_features(base_feature, &features_map)
                .into_iter()
                .map(|f| Symbol::intern(f))
        })
        .collect();

    // State gathered for "tied features" check.
    let mut enabled_disabled_features = FxHashMap::default();

    // Add enabled and remove disabled features.
    parse_rust_feature_list(
        sess,
        &sess.opts.cg.target_feature,
        /* err_callback */
        |feature| {
            sess.dcx().emit_warn(diagnostics::UnknownCTargetFeaturePrefix { feature });
        },
        |base_feature, new_features, enable| {
            match features_map.get(base_feature) {
                None => {
                    // This is definitely not a valid Rust feature name. We do not add it to
                    // `features`. Maybe it is a backend feature name? If so, give a better error
                    // message.
                    let rust_feature = sess.target.rust_target_features().iter().find_map(
                        |&(rust_feature, _, _)| {
                            let backend_features = to_backend_features(rust_feature);
                            if backend_features.contains(&base_feature)
                                && !backend_features.contains(&rust_feature)
                            {
                                Some(rust_feature)
                            } else {
                                None
                            }
                        },
                    );
                    let unknown_feature = if let Some(rust_feature) = rust_feature {
                        diagnostics::UnknownCTargetFeature {
                            feature: base_feature,
                            rust_feature: diagnostics::PossibleFeature::Some { rust_feature },
                        }
                    } else {
                        diagnostics::UnknownCTargetFeature {
                            feature: base_feature,
                            rust_feature: diagnostics::PossibleFeature::None,
                        }
                    };
                    sess.dcx().emit_warn(unknown_feature);
                }
                Some((stability, _)) => {
                    let new_features = new_features.unwrap();
                    // Add feature to our set -- only if it is actually a recognized feature.
                    // Iteration order is irrelevant since this only influences an `FxHashMap`.
                    #[allow(rustc::potential_query_instability)]
                    enabled_disabled_features.extend(new_features.iter().map(|&s| (s, enable)));

                    // Iteration order is irrelevant since this only influences an `UnordSet`.
                    #[allow(rustc::potential_query_instability)]
                    if enable {
                        features.extend(new_features.into_iter().map(|f| Symbol::intern(f)));
                    } else {
                        // Remove `new_features` from `features`.
                        for new in new_features {
                            features.remove(&Symbol::intern(new));
                        }
                    }

                    // Check feature stability.
                    if let Stability::InternalOnly { reason, hard_error } = stability {
                        let diag = diagnostics::InternalOnlyCTargetFeature {
                            feature: base_feature,
                            enabled: if enable { "enabled" } else { "disabled" },
                            reason,
                            future_compat_note: !hard_error,
                        };

                        if *hard_error {
                            sess.dcx().emit_err(diag);
                        } else {
                            sess.dcx().emit_warn(diag);
                        }
                    } else if stability.requires_nightly(/* in_cfg */ false).is_some() {
                        // An unstable feature. Warn about using it. It makes little sense
                        // to hard-error here since we just warn about fully unknown
                        // features above.
                        let note = if stability.is_cfg_stable_toggle_unstable() {
                            "this feature is allowed in cfg but unstable otherwise"
                        } else {
                            "this feature is not stably supported"
                        };
                        sess.dcx().emit_warn(diagnostics::UnstableCTargetFeature {
                            feature: base_feature,
                            note,
                        });
                    }
                }
            }
        },
    );

    if let Some(f) = check_tied_features(sess, &enabled_disabled_features) {
        sess.dcx().emit_err(diagnostics::TargetFeatureDisableOrEnable {
            features: f,
            span: None,
            missing_features: None,
        });
    }

    features
}

/// Given a map from target_features to whether they are enabled or disabled, ensure only valid
/// combinations are allowed. Returns `Some` if a violation is found.
pub fn check_tied_features(
    sess: &Session,
    features: &FxHashMap<&str, bool>,
) -> Option<&'static [&'static str]> {
    if !features.is_empty() {
        for tied in sess.target.tied_target_features() {
            // Tied features must be set to the same value, or not set at all
            let mut tied_iter = tied.iter();
            let enabled = features.get(tied_iter.next().unwrap());
            if tied_iter.any(|f| enabled != features.get(f)) {
                return Some(tied);
            }
        }
    }
    None
}

/// Translates the target spec `features` field into a backend target feature list.
///
/// `extend_backend_features` extends the set of backend features (assumed to be in mutable state
/// accessible by that closure) to enable/disable the given Rust feature name.
pub fn target_spec_to_backend_features<'a>(
    sess: &'a Session,
    mut extend_backend_features: impl FnMut(&'a str, /* enable */ bool),
) {
    // This check handles SM versions that defaults (by LLVM) to unsupported (by Rust) PTX ISA versions.
    // sm_70, sm_72 and sm_75 defaults to PTX ISA versions with major version 6, while sm_80 default to 7.0
    if sess.target.arch == Arch::Nvptx64
        && matches!(
            sess.opts.cg.target_cpu.as_deref(),
            None | Some("sm_70") | Some("sm_72") | Some("sm_75")
        )
    {
        extend_backend_features("ptx70", true);
    }

    // Compute implied features
    parse_rust_feature_list(
        sess,
        &sess.target.features,
        /* err_callback */
        |feature| {
            panic!("Target spec contains invalid feature {feature} (missing `+`/`-` prefix)");
        },
        |base_feature, new_features, enable| {
            // FIXME emit an error for unknown features in the target spec like
            // internal_target_features would for -Ctarget-feature.
            let new_features =
                new_features.unwrap_or_else(|| FxHashSet::from_iter(std::iter::once(base_feature)));
            for new_feature in UnordSet::from(new_features).to_sorted_stable_ord().iter() {
                extend_backend_features(new_feature, enable);
            }
        },
    );
}

/// Translates the `-Ctarget-feature` flag into a backend target feature list.
///
/// `extend_backend_features` extends the set of backend features (assumed to be in mutable state
/// accessible by that closure) to enable/disable the given Rust feature name.
pub fn flag_to_backend_features<'a>(
    sess: &'a Session,
    mut extend_backend_features: impl FnMut(&'a str, /* enable */ bool),
) {
    parse_rust_feature_list(
        sess,
        &sess.opts.cg.target_feature,
        /* err_callback */
        |_feature| {
            // Errors are already emitted in `internal_target_features`; avoid duplicates.
        },
        |base_feature, new_features, enable| {
            // Forward unknown features to the backend as that's what we have always done.
            let new_features =
                new_features.unwrap_or_else(|| FxHashSet::from_iter(std::iter::once(base_feature)));
            for new_feature in UnordSet::from(new_features).to_sorted_stable_ord().iter() {
                extend_backend_features(new_feature, enable);
            }
        },
    );
}

/// Computes the backend target features to be added to account for retpoline flags.
/// Used by both LLVM and GCC since their target features are, conveniently, the same.
pub fn retpoline_features_by_flags(sess: &Session, features: &mut Vec<String>) {
    // -Tretpoline without -Tretpoline-external-thunk enables
    // retpoline-indirect-branches and retpoline-indirect-calls target features
    let target_opts = &sess.opts.target_opts;
    if target_opts.retpoline && !target_opts.retpoline_external_thunk {
        features.push("+retpoline-indirect-branches".into());
        features.push("+retpoline-indirect-calls".into());
    }
    // -Tretpoline-external-thunk (maybe, with -Tretpoline too) enables
    // retpoline-external-thunk, retpoline-indirect-branches and
    // retpoline-indirect-calls target features
    if target_opts.retpoline_external_thunk {
        features.push("+retpoline-external-thunk".into());
        features.push("+retpoline-indirect-branches".into());
        features.push("+retpoline-indirect-calls".into());
    }
}

/// Computes the backend target features to be added to account for sanitizer flags.
pub fn sanitizer_features_by_flags(sess: &Session, features: &mut Vec<String>) {
    // It's intentional that this is done only for non-kernel version of hwaddress. This matches
    // clang behavior.
    if sess.sanitizers().contains(SanitizerSet::HWADDRESS) {
        features.push("+tagged-globals".into());
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        rust_target_features: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            if tcx.sess.opts.actually_rustdoc {
                // HACK: rustdoc would like to pretend that we have all the target features, so we
                // have to merge all the lists into one. To ensure an unstable target never prevents
                // a stable one from working, we merge the stability info of all instances of the
                // same target feature name, with the "most stable" taking precedence. And then we
                // hope that this doesn't cause issues anywhere else in the compiler...
                let mut result: UnordMap<String, Stability> = Default::default();
                for (name, stability) in rustc_target::target_features::all_rust_features() {
                    use std::collections::hash_map::Entry;
                    match result.entry(name.to_owned()) {
                        Entry::Vacant(vacant_entry) => {
                            vacant_entry.insert(stability);
                        }
                        Entry::Occupied(mut occupied_entry) => {
                            // Merge the two stabilities, "more stable" taking precedence.
                            match (occupied_entry.get(), stability) {
                                (Stability::Stable, _)
                                | (
                                    Stability::Unstable { .. },
                                    Stability::Unstable { .. } | Stability::InternalOnly { .. },
                                )
                                | (
                                    Stability::InternalOnly { .. },
                                    Stability::InternalOnly { .. },
                                ) => {
                                    // The stability in the entry is at least as good as the new
                                    // one, just keep it.
                                }
                                _ => {
                                    // Overwrite stability.
                                    occupied_entry.insert(stability);
                                }
                            }
                        }
                    }
                }
                result
            } else {
                tcx.sess
                    .target
                    .rust_target_features()
                    .iter()
                    .map(|(feat, stab, _)| (feat.to_string(), *stab))
                    .collect()
            }
        },
        implied_target_features: |tcx, feature: Symbol| {
            if tcx.sess.opts.actually_rustdoc {
                // We can't handle implication when we are mixing all targets.
                return vec![feature];
            }
            let features_map = tcx.sess.target.rust_target_features_map();
            let feature = feature.as_str();
            UnordSet::from(tcx.sess.target.implied_target_features(feature, &features_map))
                .into_sorted_stable_ord()
                .into_iter()
                .map(|s| Symbol::intern(s))
                .collect()
        },
        asm_target_features,
        ..*providers
    }
}
