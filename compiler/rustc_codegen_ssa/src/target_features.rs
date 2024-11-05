use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::InstructionSetAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_middle::middle::codegen_fn_attrs::TargetFeature;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::parse::feature_err;
use rustc_span::Span;
use rustc_span::symbol::{Symbol, sym};
use rustc_target::target_features::{self, Stability};

use crate::errors;

/// Compute the enabled target features from the `#[target_feature]` function attribute.
/// Enabled target features are added to `target_features`.
pub(crate) fn from_target_feature_attr(
    tcx: TyCtxt<'_>,
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
    let mut added_target_features = Vec::new();
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
        added_target_features.extend(value.as_str().split(',').filter_map(|feature| {
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
                return None;
            };

            // Only allow target features whose feature gates have been enabled.
            let allowed = match stability {
                Stability::Forbidden { .. } => false,
                Stability::Stable => true,
                Stability::Unstable(name) => rust_features.enabled(*name),
            };
            if !allowed {
                match stability {
                    Stability::Stable => unreachable!(),
                    &Stability::Unstable(lang_feature_name) => {
                        feature_err(
                            &tcx.sess,
                            lang_feature_name,
                            item.span(),
                            format!("the target feature `{feature}` is currently unstable"),
                        )
                        .emit();
                    }
                    Stability::Forbidden { reason } => {
                        tcx.dcx().emit_err(errors::ForbiddenTargetFeatureAttr {
                            span: item.span(),
                            feature,
                            reason,
                        });
                    }
                }
            }
            Some(Symbol::intern(feature))
        }));
    }

    // Add explicit features
    target_features.extend(
        added_target_features.iter().copied().map(|name| TargetFeature { name, implied: false }),
    );

    // Add implied features
    let mut implied_target_features = UnordSet::new();
    for feature in added_target_features.iter() {
        implied_target_features.extend(tcx.implied_target_features(*feature).clone());
    }
    for feature in added_target_features.iter() {
        implied_target_features.remove(feature);
    }
    target_features.extend(
        implied_target_features
            .into_sorted_stable_ord()
            .iter()
            .copied()
            .map(|name| TargetFeature { name, implied: true }),
    )
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
                // rustdoc needs to be able to document functions that use all the features, so
                // whitelist them all
                rustc_target::target_features::all_rust_features()
                    .map(|(a, b)| (a.to_string(), b))
                    .collect()
            } else {
                tcx.sess
                    .target
                    .rust_target_features()
                    .iter()
                    .map(|&(a, b, _)| (a.to_string(), b))
                    .collect()
            }
        },
        implied_target_features: |tcx, feature| {
            UnordSet::from(tcx.sess.target.implied_target_features(std::iter::once(feature)))
                .into_sorted_stable_ord()
        },
        asm_target_features,
        ..*providers
    }
}
