//! Detecting lib features (i.e., features that are not lang features).
//!
//! These are declared using stability attributes (e.g., `#[stable (..)]` and `#[unstable (..)]`),
//! but are not declared in one single location (unlike lang features), which means we need to
//! collect them instead.

use rustc_attr::VERSION_PLACEHOLDER;
use rustc_hir::Attribute;
use rustc_hir::intravisit::Visitor;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::lib_features::{FeatureStability, LibFeatures};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};

use crate::errors::{FeaturePreviouslyDeclared, FeatureStableTwice};

struct LibFeatureCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    lib_features: LibFeatures,
}

impl<'tcx> LibFeatureCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LibFeatureCollector<'tcx> {
        LibFeatureCollector { tcx, lib_features: LibFeatures::default() }
    }

    fn extract(&self, attr: &Attribute) -> Option<(Symbol, FeatureStability, Span)> {
        let stab_attrs = [
            sym::stable,
            sym::unstable,
            sym::rustc_const_stable,
            sym::rustc_const_unstable,
            sym::rustc_default_body_unstable,
        ];

        // Find a stability attribute: one of #[stable(…)], #[unstable(…)],
        // #[rustc_const_stable(…)], #[rustc_const_unstable(…)] or #[rustc_default_body_unstable].
        if let Some(stab_attr) = stab_attrs.iter().find(|stab_attr| attr.has_name(**stab_attr)) {
            if let Some(metas) = attr.meta_item_list() {
                let mut feature = None;
                let mut since = None;
                for meta in metas {
                    if let Some(mi) = meta.meta_item() {
                        // Find the `feature = ".."` meta-item.
                        match (mi.name_or_empty(), mi.value_str()) {
                            (sym::feature, val) => feature = val,
                            (sym::since, val) => since = val,
                            _ => {}
                        }
                    }
                }

                if let Some(s) = since
                    && s.as_str() == VERSION_PLACEHOLDER
                {
                    since = Some(sym::env_CFG_RELEASE);
                }

                if let Some(feature) = feature {
                    // This additional check for stability is to make sure we
                    // don't emit additional, irrelevant errors for malformed
                    // attributes.
                    let is_unstable = matches!(
                        *stab_attr,
                        sym::unstable
                            | sym::rustc_const_unstable
                            | sym::rustc_default_body_unstable
                    );
                    if is_unstable {
                        return Some((feature, FeatureStability::Unstable, attr.span));
                    }
                    if let Some(since) = since {
                        return Some((feature, FeatureStability::AcceptedSince(since), attr.span));
                    }
                }
                // We need to iterate over the other attributes, because
                // `rustc_const_unstable` is not mutually exclusive with
                // the other stability attributes, so we can't just `break`
                // here.
            }
        }

        None
    }

    fn collect_feature(&mut self, feature: Symbol, stability: FeatureStability, span: Span) {
        let existing_stability = self.lib_features.stability.get(&feature).cloned();

        match (stability, existing_stability) {
            (_, None) => {
                self.lib_features.stability.insert(feature, (stability, span));
            }
            (
                FeatureStability::AcceptedSince(since),
                Some((FeatureStability::AcceptedSince(prev_since), _)),
            ) => {
                if prev_since != since {
                    self.tcx.dcx().emit_err(FeatureStableTwice {
                        span,
                        feature,
                        since,
                        prev_since,
                    });
                }
            }
            (FeatureStability::AcceptedSince(_), Some((FeatureStability::Unstable, _))) => {
                self.tcx.dcx().emit_err(FeaturePreviouslyDeclared {
                    span,
                    feature,
                    declared: "stable",
                    prev_declared: "unstable",
                });
            }
            (FeatureStability::Unstable, Some((FeatureStability::AcceptedSince(_), _))) => {
                self.tcx.dcx().emit_err(FeaturePreviouslyDeclared {
                    span,
                    feature,
                    declared: "unstable",
                    prev_declared: "stable",
                });
            }
            // duplicate `unstable` feature is ok.
            (FeatureStability::Unstable, Some((FeatureStability::Unstable, _))) => {}
        }
    }
}

impl<'tcx> Visitor<'tcx> for LibFeatureCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_attribute(&mut self, attr: &'tcx Attribute) {
        if let Some((feature, stable, span)) = self.extract(attr) {
            self.collect_feature(feature, stable, span);
        }
    }
}

fn lib_features(tcx: TyCtxt<'_>, LocalCrate: LocalCrate) -> LibFeatures {
    // If `staged_api` is not enabled then we aren't allowed to define lib
    // features; there is no point collecting them.
    if !tcx.features().staged_api() {
        return LibFeatures::default();
    }

    let mut collector = LibFeatureCollector::new(tcx);
    tcx.hir().walk_attributes(&mut collector);
    collector.lib_features
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.lib_features = lib_features;
}
