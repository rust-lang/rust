// Detecting lib features (i.e., features that are not lang features).
//
// These are declared using stability attributes (e.g., `#[stable (..)]`
// and `#[unstable (..)]`), but are not declared in one single location
// (unlike lang features), which means we need to collect them instead.

use ty::TyCtxt;
use syntax::symbol::Symbol;
use syntax::ast::{Attribute, MetaItem, MetaItemKind};
use syntax_pos::Span;
use hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use errors::DiagnosticId;

pub struct LibFeatures {
    // A map from feature to stabilisation version.
    pub stable: FxHashMap<Symbol, Symbol>,
    pub unstable: FxHashSet<Symbol>,
}

impl LibFeatures {
    fn new() -> LibFeatures {
        LibFeatures {
            stable: Default::default(),
            unstable: Default::default(),
        }
    }

    pub fn to_vec(&self) -> Vec<(Symbol, Option<Symbol>)> {
        let mut all_features: Vec<_> = self.stable.iter().map(|(f, s)| (*f, Some(*s)))
            .chain(self.unstable.iter().map(|f| (*f, None)))
            .collect();
        all_features.sort_unstable_by_key(|f| f.0.as_str());
        all_features
    }
}

pub struct LibFeatureCollector<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    lib_features: LibFeatures,
}

impl<'a, 'tcx> LibFeatureCollector<'a, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> LibFeatureCollector<'a, 'tcx> {
        LibFeatureCollector {
            tcx,
            lib_features: LibFeatures::new(),
        }
    }

    fn extract(&self, attr: &Attribute) -> Option<(Symbol, Option<Symbol>, Span)> {
        let stab_attrs = vec!["stable", "unstable", "rustc_const_unstable"];

        // Find a stability attribute (i.e., `#[stable (..)]`, `#[unstable (..)]`,
        // `#[rustc_const_unstable (..)]`).
        if let Some(stab_attr) = stab_attrs.iter().find(|stab_attr| {
            attr.check_name(stab_attr)
        }) {
            let meta_item = attr.meta();
            if let Some(MetaItem { node: MetaItemKind::List(ref metas), .. }) = meta_item {
                let mut feature = None;
                let mut since = None;
                for meta in metas {
                    if let Some(mi) = meta.meta_item() {
                        // Find the `feature = ".."` meta-item.
                        match (&*mi.name().as_str(), mi.value_str()) {
                            ("feature", val) => feature = val,
                            ("since", val) => since = val,
                            _ => {}
                        }
                    }
                }
                if let Some(feature) = feature {
                    // This additional check for stability is to make sure we
                    // don't emit additional, irrelevant errors for malformed
                    // attributes.
                    if *stab_attr != "stable" || since.is_some() {
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
        let already_in_unstable = self.lib_features.unstable.contains(&feature);

        match (since, already_in_stable, already_in_unstable) {
            (Some(since), _, false) => {
                if let Some(prev_since) = self.lib_features.stable.get(&feature) {
                    if *prev_since != since {
                        let msg = format!(
                            "feature `{}` is declared stable since {}, \
                             but was previously declared stable since {}",
                            feature,
                            since,
                            prev_since,
                        );
                        self.tcx.sess.struct_span_err_with_code(span, &msg,
                            DiagnosticId::Error("E0711".into())).emit();
                        return;
                    }
                }

                self.lib_features.stable.insert(feature, since);
            }
            (None, false, _) => {
                self.lib_features.unstable.insert(feature);
            }
            (Some(_), _, true) | (None, true, _) => {
                let msg = format!(
                    "feature `{}` is declared {}, but was previously declared {}",
                    feature,
                    if since.is_some() { "stable" } else { "unstable" },
                    if since.is_none() { "stable" } else { "unstable" },
                );
                self.tcx.sess.struct_span_err_with_code(span, &msg,
                    DiagnosticId::Error("E0711".into())).emit();
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for LibFeatureCollector<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_attribute(&mut self, attr: &'tcx Attribute) {
        if let Some((feature, stable, span)) = self.extract(attr) {
            self.collect_feature(feature, stable, span);
        }
    }
}

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> LibFeatures {
    let mut collector = LibFeatureCollector::new(tcx);
    intravisit::walk_crate(&mut collector, tcx.hir().krate());
    collector.lib_features
}
