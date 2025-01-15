use rustc_attr_data_structures::InstructionSetAttr;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_middle::middle::codegen_fn_attrs::TargetFeature;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::features::StabilityExt;
use rustc_session::lint::builtin::AARCH64_SOFTFLOAT_NEON;
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};
use rustc_target::target_features::{self, Stability};

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
            if let Err(reason) = stability.is_toggle_permitted(tcx.sess) {
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
                    // Here we do assume that LLVM doesn't add even more implied features
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
