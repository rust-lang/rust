use rustc_attr_data_structures::InstructionSetAttr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_middle::middle::codegen_fn_attrs::TargetFeature;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::builtin::AARCH64_SOFTFLOAT_NEON;
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};
use rustc_target::target_features::{
    self, RUSTC_SPECIAL_FEATURES, RUSTC_SPECIFIC_FEATURES, Stability,
};
use smallvec::SmallVec;

use crate::errors;

/// Compute the enabled target features from the `#[target_feature]` function attribute.
/// Enabled target features are added to `target_features`.
pub(crate) fn from_target_feature_attr(
    tcx: TyCtxt<'_>,
    did: LocalDefId,
    attr: &hir::Attribute,
    rust_target_features: &UnordMap<String, target_features::Stability>,
    target_features: &mut Vec<TargetFeature>,
) {
    let Some(list) = attr.meta_item_list() else { return };
    let bad_item = |span| {
        let msg = "malformed `target_feature` attribute input";
        let code = "enable = \"..\"";
        tcx.dcx()
            .struct_span_err(span, msg)
            .with_span_suggestion(span, "must be of the form", code, Applicability::HasPlaceholders)
            .emit();
    };
    let rust_features = tcx.features();
    let abi_feature_constraints = tcx.sess.target.abi_required_features();
    for item in list {
        // Only `enable = ...` is accepted in the meta-item list.
        if !item.has_name(sym::enable) {
            bad_item(item.span());
            continue;
        }

        // Must be of the form `enable = "..."` (a string).
        let Some(value) = item.value_str() else {
            bad_item(item.span());
            continue;
        };

        // We allow comma separation to enable multiple features.
        for feature in value.as_str().split(',') {
            let Some(stability) = rust_target_features.get(feature) else {
                let msg = format!("the feature named `{feature}` is not valid for this target");
                let mut err = tcx.dcx().struct_span_err(item.span(), msg);
                err.span_label(item.span(), format!("`{feature}` is not valid for this target"));
                if let Some(stripped) = feature.strip_prefix('+') {
                    let valid = rust_target_features.contains_key(stripped);
                    if valid {
                        err.help("consider removing the leading `+` in the feature name");
                    }
                }
                err.emit();
                continue;
            };

            // Only allow target features whose feature gates have been enabled
            // and which are permitted to be toggled.
            if let Err(reason) = stability.toggle_allowed() {
                tcx.dcx().emit_err(errors::ForbiddenTargetFeatureAttr {
                    span: item.span(),
                    feature,
                    reason,
                });
            } else if let Some(nightly_feature) = stability.requires_nightly()
                && !rust_features.enabled(nightly_feature)
            {
                feature_err(
                    &tcx.sess,
                    nightly_feature,
                    item.span(),
                    format!("the target feature `{feature}` is currently unstable"),
                )
                .emit();
            } else {
                // Add this and the implied features.
                let feature_sym = Symbol::intern(feature);
                for &name in tcx.implied_target_features(feature_sym) {
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
                            if tcx.sess.target.arch == "aarch64" && name.as_str() == "neon" {
                                tcx.emit_node_span_lint(
                                    AARCH64_SOFTFLOAT_NEON,
                                    tcx.local_def_id_to_hir_id(did),
                                    item.span(),
                                    errors::Aarch64SoftfloatNeon,
                                );
                            } else {
                                tcx.dcx().emit_err(errors::ForbiddenTargetFeatureAttr {
                                    span: item.span(),
                                    feature: name.as_str(),
                                    reason: "this feature is incompatible with the target ABI",
                                });
                            }
                        }
                    }
                    target_features.push(TargetFeature { name, implied: name != feature_sym })
                }
            }
        }
    }
}

/// Computes the set of target features used in a function for the purposes of
/// inline assembly.
fn asm_target_features(tcx: TyCtxt<'_>, did: DefId) -> &FxIndexSet<Symbol> {
    let mut target_features = tcx.sess.unstable_target_features.clone();
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
            tcx.dcx().emit_err(errors::TargetFeatureSafeTrait {
                span: attr_span,
                def: tcx.def_span(id),
            });
        }
    }
}

/// Utility function for a codegen backend to compute `cfg(target_feature)`, or more specifically,
/// to populate `sess.unstable_target_features` and `sess.target_features` (these are the first and
/// 2nd component of the return value, respectively).
///
/// `target_feature_flag` is the value of `-Ctarget-feature` (giving the caller a chance to override it).
/// `target_base_has_feature` should check whether the given feature (a Rust feature name!) is enabled
/// in the "base" target machine, i.e., without applying `-Ctarget-feature`.
///
/// We do not have to worry about RUSTC_SPECIFIC_FEATURES here, those are handled elsewhere.
pub fn cfg_target_feature(
    sess: &Session,
    target_feature_flag: &str,
    mut is_feature_enabled: impl FnMut(&str) -> bool,
) -> (Vec<Symbol>, Vec<Symbol>) {
    // Compute which of the known target features are enabled in the 'base' target machine. We only
    // consider "supported" features; "forbidden" features are not reflected in `cfg` as of now.
    let mut features: FxHashSet<Symbol> = sess
        .target
        .rust_target_features()
        .iter()
        .filter(|(feature, _, _)| {
            // Skip checking special features, those are not known to the backend.
            if RUSTC_SPECIAL_FEATURES.contains(feature) {
                return true;
            }
            is_feature_enabled(feature)
        })
        .map(|(feature, _, _)| Symbol::intern(feature))
        .collect();

    // Add enabled and remove disabled features.
    for (enabled, feature) in
        target_feature_flag.split(',').filter_map(|s| match s.chars().next() {
            Some('+') => Some((true, Symbol::intern(&s[1..]))),
            Some('-') => Some((false, Symbol::intern(&s[1..]))),
            _ => None,
        })
    {
        if enabled {
            // Also add all transitively implied features.

            // We don't care about the order in `features` since the only thing we use it for is the
            // `features.contains` below.
            #[allow(rustc::potential_query_instability)]
            features.extend(
                sess.target
                    .implied_target_features(feature.as_str())
                    .iter()
                    .map(|s| Symbol::intern(s)),
            );
        } else {
            // Remove transitively reverse-implied features.

            // We don't care about the order in `features` since the only thing we use it for is the
            // `features.contains` below.
            #[allow(rustc::potential_query_instability)]
            features.retain(|f| {
                if sess.target.implied_target_features(f.as_str()).contains(&feature.as_str()) {
                    // If `f` if implies `feature`, then `!feature` implies `!f`, so we have to
                    // remove `f`. (This is the standard logical contraposition principle.)
                    false
                } else {
                    // We can keep `f`.
                    true
                }
            });
        }
    }

    // Filter enabled features based on feature gates.
    let f = |allow_unstable| {
        sess.target
            .rust_target_features()
            .iter()
            .filter_map(|(feature, gate, _)| {
                // The `allow_unstable` set is used by rustc internally to determine which target
                // features are truly available, so we want to return even perma-unstable
                // "forbidden" features.
                if allow_unstable
                    || (gate.in_cfg()
                        && (sess.is_nightly_build() || gate.requires_nightly().is_none()))
                {
                    Some(Symbol::intern(feature))
                } else {
                    None
                }
            })
            .filter(|feature| features.contains(&feature))
            .collect()
    };

    (f(true), f(false))
}

/// Given a map from target_features to whether they are enabled or disabled, ensure only valid
/// combinations are allowed.
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

/// Translates the `-Ctarget-feature` flag into a backend target feature list.
///
/// `to_backend_features` converts a Rust feature name into a list of backend feature names; this is
/// used for diagnostic purposes only.
///
/// `extend_backend_features` extends the set of backend features (assumed to be in mutable state
/// accessible by that closure) to enable/disable the given Rust feature name.
pub fn flag_to_backend_features<'a, const N: usize>(
    sess: &'a Session,
    diagnostics: bool,
    to_backend_features: impl Fn(&'a str) -> SmallVec<[&'a str; N]>,
    mut extend_backend_features: impl FnMut(&'a str, /* enable */ bool),
) {
    let known_features = sess.target.rust_target_features();

    // Compute implied features
    let mut rust_features = vec![];
    for feature in sess.opts.cg.target_feature.split(',') {
        if let Some(feature) = feature.strip_prefix('+') {
            rust_features.extend(
                UnordSet::from(sess.target.implied_target_features(feature))
                    .to_sorted_stable_ord()
                    .iter()
                    .map(|&&s| (true, s)),
            )
        } else if let Some(feature) = feature.strip_prefix('-') {
            // FIXME: Why do we not remove implied features on "-" here?
            // We do the equivalent above in `target_config`.
            // See <https://github.com/rust-lang/rust/issues/134792>.
            rust_features.push((false, feature));
        } else if !feature.is_empty() {
            if diagnostics {
                sess.dcx().emit_warn(errors::UnknownCTargetFeaturePrefix { feature });
            }
        }
    }
    // Remove features that are meant for rustc, not the backend.
    rust_features.retain(|(_, feature)| {
        // Retain if it is not a rustc feature
        !RUSTC_SPECIFIC_FEATURES.contains(feature)
    });

    // Check feature validity.
    if diagnostics {
        let mut featsmap = FxHashMap::default();

        for &(enable, feature) in &rust_features {
            let feature_state = known_features.iter().find(|&&(v, _, _)| v == feature);
            match feature_state {
                None => {
                    // This is definitely not a valid Rust feature name. Maybe it is a backend feature name?
                    // If so, give a better error message.
                    let rust_feature = known_features.iter().find_map(|&(rust_feature, _, _)| {
                        let backend_features = to_backend_features(rust_feature);
                        if backend_features.contains(&feature)
                            && !backend_features.contains(&rust_feature)
                        {
                            Some(rust_feature)
                        } else {
                            None
                        }
                    });
                    let unknown_feature = if let Some(rust_feature) = rust_feature {
                        errors::UnknownCTargetFeature {
                            feature,
                            rust_feature: errors::PossibleFeature::Some { rust_feature },
                        }
                    } else {
                        errors::UnknownCTargetFeature {
                            feature,
                            rust_feature: errors::PossibleFeature::None,
                        }
                    };
                    sess.dcx().emit_warn(unknown_feature);
                }
                Some((_, stability, _)) => {
                    if let Err(reason) = stability.toggle_allowed() {
                        sess.dcx().emit_warn(errors::ForbiddenCTargetFeature {
                            feature,
                            enabled: if enable { "enabled" } else { "disabled" },
                            reason,
                        });
                    } else if stability.requires_nightly().is_some() {
                        // An unstable feature. Warn about using it. It makes little sense
                        // to hard-error here since we just warn about fully unknown
                        // features above.
                        sess.dcx().emit_warn(errors::UnstableCTargetFeature { feature });
                    }
                }
            }

            // FIXME(nagisa): figure out how to not allocate a full hashset here.
            featsmap.insert(feature, enable);
        }

        if let Some(f) = check_tied_features(sess, &featsmap) {
            sess.dcx().emit_err(errors::TargetFeatureDisableOrEnable {
                features: f,
                span: None,
                missing_features: None,
            });
        }
    }

    // Add this to the backend features.
    for (enable, feature) in rust_features {
        extend_backend_features(feature, enable);
    }
}

/// Computes the backend target features to be added to account for retpoline flags.
/// Used by both LLVM and GCC since their target features are, conveniently, the same.
pub fn retpoline_features_by_flags(sess: &Session, features: &mut Vec<String>) {
    // -Zretpoline without -Zretpoline-external-thunk enables
    // retpoline-indirect-branches and retpoline-indirect-calls target features
    let unstable_opts = &sess.opts.unstable_opts;
    if unstable_opts.retpoline && !unstable_opts.retpoline_external_thunk {
        features.push("+retpoline-indirect-branches".into());
        features.push("+retpoline-indirect-calls".into());
    }
    // -Zretpoline-external-thunk (maybe, with -Zretpoline too) enables
    // retpoline-external-thunk, retpoline-indirect-branches and
    // retpoline-indirect-calls target features
    if unstable_opts.retpoline_external_thunk {
        features.push("+retpoline-external-thunk".into());
        features.push("+retpoline-indirect-branches".into());
        features.push("+retpoline-indirect-calls".into());
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
                                    Stability::Unstable { .. } | Stability::Forbidden { .. },
                                )
                                | (Stability::Forbidden { .. }, Stability::Forbidden { .. }) => {
                                    // The stability in the entry is at least as good as the new one, just keep it.
                                }
                                _ => {
                                    // Overwrite stabilite.
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
                    .map(|(a, b, _)| (a.to_string(), *b))
                    .collect()
            }
        },
        implied_target_features: |tcx, feature: Symbol| {
            let feature = feature.as_str();
            UnordSet::from(tcx.sess.target.implied_target_features(feature))
                .into_sorted_stable_ord()
                .into_iter()
                .map(|s| Symbol::intern(s))
                .collect()
        },
        asm_target_features,
        ..*providers
    }
}
