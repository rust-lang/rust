//! Detecting lib features (i.e., features that are not lang features).
//!
//! These are declared using stability attributes (e.g., `#[stable (..)]` and `#[unstable (..)]`),
//! but are not declared in one single location (unlike lang features), which means we need to
//! collect them instead.

use rustc_hir::attrs::AttributeKind;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{Attribute, StabilityLevel, StableSince};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::lib_features::{FeatureStability, LibFeatures};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_span::{Span, Symbol, sym};

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
        let (feature, level, span) = match attr {
            Attribute::Parsed(AttributeKind::Stability { stability, span }) => {
                (stability.feature, stability.level, *span)
            }
            Attribute::Parsed(AttributeKind::ConstStability { stability, span }) => {
                (stability.feature, stability.level, *span)
            }
            Attribute::Parsed(AttributeKind::BodyStability { stability, span }) => {
                (stability.feature, stability.level, *span)
            }
            _ => return None,
        };

        let feature_stability = match level {
            StabilityLevel::Unstable { old_name, .. } => FeatureStability::Unstable { old_name },
            StabilityLevel::Stable { since, .. } => FeatureStability::AcceptedSince(match since {
                StableSince::Version(v) => Symbol::intern(&v.to_string()),
                StableSince::Current => sym::env_CFG_RELEASE,
                StableSince::Err(_) => return None,
            }),
        };

        Some((feature, feature_stability, span))
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
            (FeatureStability::AcceptedSince(_), Some((FeatureStability::Unstable { .. }, _))) => {
                self.tcx.dcx().emit_err(FeaturePreviouslyDeclared {
                    span,
                    feature,
                    declared: "stable",
                    prev_declared: "unstable",
                });
            }
            (FeatureStability::Unstable { .. }, Some((FeatureStability::AcceptedSince(_), _))) => {
                self.tcx.dcx().emit_err(FeaturePreviouslyDeclared {
                    span,
                    feature,
                    declared: "unstable",
                    prev_declared: "stable",
                });
            }
            // duplicate `unstable` feature is ok.
            (FeatureStability::Unstable { .. }, Some((FeatureStability::Unstable { .. }, _))) => {}
        }
    }
}

impl<'tcx> Visitor<'tcx> for LibFeatureCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
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
    tcx.hir_walk_attributes(&mut collector);
    collector.lib_features
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.lib_features = lib_features;
}
