//! Detecting lib features (i.e., features that are not lang features).
//!
//! These are declared using stability attributes (e.g., `#[stable (..)]` and `#[unstable (..)]`),
//! but are not declared in one single location (unlike lang features), which means we need to
//! collect them instead.

use rustc_ast::Attribute;
use rustc_attr::{rust_version_symbol, VERSION_PLACEHOLDER};
use rustc_hir::intravisit::Visitor;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::lib_features::LibFeatures;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

use crate::errors::{FeaturePreviouslyDeclared, FeatureStableTwice};

fn new_lib_features() -> LibFeatures {
    LibFeatures { stable: Default::default(), unstable: Default::default() }
}

pub struct LibFeatureCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    lib_features: LibFeatures,
}

impl<'tcx> LibFeatureCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LibFeatureCollector<'tcx> {
        LibFeatureCollector { tcx, lib_features: new_lib_features() }
    }

    fn extract(&self, attr: &Attribute) -> Option<(Symbol, Option<Symbol>, Span)> {
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

                if let Some(s) = since && s.as_str() == VERSION_PLACEHOLDER {
                    since = Some(rust_version_symbol());
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
                    if since.is_some() || is_unstable {
                        return Some((feature, since, attr.span));
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

    fn collect_feature(&mut self, feature: Symbol, since: Option<Symbol>, span: Span) {
        let already_in_stable = self.lib_features.stable.contains_key(&feature);
        let already_in_unstable = self.lib_features.unstable.contains_key(&feature);

        match (since, already_in_stable, already_in_unstable) {
            (Some(since), _, false) => {
                if let Some((prev_since, _)) = self.lib_features.stable.get(&feature) {
                    if *prev_since != since {
                        self.tcx.sess.emit_err(FeatureStableTwice {
                            span,
                            feature,
                            since,
                            prev_since: *prev_since,
                        });
                        return;
                    }
                }

                self.lib_features.stable.insert(feature, (since, span));
            }
            (None, false, _) => {
                self.lib_features.unstable.insert(feature, span);
            }
            (Some(_), _, true) | (None, true, _) => {
                let declared = if since.is_some() { "stable" } else { "unstable" };
                let prev_declared = if since.is_none() { "stable" } else { "unstable" };
                self.tcx.sess.emit_err(FeaturePreviouslyDeclared {
                    span,
                    feature,
                    declared,
                    prev_declared,
                });
            }
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

fn lib_features(tcx: TyCtxt<'_>, (): ()) -> LibFeatures {
    // If `staged_api` is not enabled then we aren't allowed to define lib
    // features; there is no point collecting them.
    if !tcx.features().staged_api {
        return new_lib_features();
    }

    let mut collector = LibFeatureCollector::new(tcx);
    tcx.hir().walk_attributes(&mut collector);
    collector.lib_features
}

pub fn provide(providers: &mut Providers) {
    providers.lib_features = lib_features;
}
