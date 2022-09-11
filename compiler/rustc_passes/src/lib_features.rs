//! Detecting lib features (i.e., features that are not lang features).
//!
//! These are declared using stability attributes (e.g., `#[stable (..)]` and `#[unstable (..)]`),
//! but are not declared in one single location (unlike lang features), which means we need to
//! collect them instead.

use rustc_ast::{Attribute, MetaItemKind};
use rustc_attr::VERSION_PLACEHOLDER;
use rustc_errors::struct_span_err;
use rustc_hir::intravisit::Visitor;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::lib_features::LibFeatures;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

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
            let meta_kind = attr.meta_kind();
            if let Some(MetaItemKind::List(ref metas)) = meta_kind {
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
                    let version = option_env!("CFG_VERSION").unwrap_or("<current>");
                    let version = version.split(' ').next().unwrap();
                    since = Some(Symbol::intern(&version));
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
                        self.span_feature_error(
                            span,
                            &format!(
                                "feature `{}` is declared stable since {}, \
                                 but was previously declared stable since {}",
                                feature, since, prev_since,
                            ),
                        );
                        return;
                    }
                }

                self.lib_features.stable.insert(feature, (since, span));
            }
            (None, false, _) => {
                self.lib_features.unstable.insert(feature, span);
            }
            (Some(_), _, true) | (None, true, _) => {
                self.span_feature_error(
                    span,
                    &format!(
                        "feature `{}` is declared {}, but was previously declared {}",
                        feature,
                        if since.is_some() { "stable" } else { "unstable" },
                        if since.is_none() { "stable" } else { "unstable" },
                    ),
                );
            }
        }
    }

    fn span_feature_error(&self, span: Span, msg: &str) {
        struct_span_err!(self.tcx.sess, span, E0711, "{}", &msg).emit();
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
    let mut collector = LibFeatureCollector::new(tcx);
    tcx.hir().walk_attributes(&mut collector);
    collector.lib_features
}

pub fn provide(providers: &mut Providers) {
    providers.lib_features = lib_features;
}
