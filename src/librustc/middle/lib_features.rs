// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Detecting lib features (i.e. features that are not lang features).
//
// These are declared using stability attributes (e.g. `#[stable (..)]`
// and `#[unstable (..)]`), but are not declared in one single location
// (unlike lang features), which means we need to collect them instead.

use ty::TyCtxt;
use syntax::symbol::Symbol;
use syntax::ast::{Attribute, MetaItem, MetaItemKind};
use syntax_pos::{Span, DUMMY_SP};
use hir;
use hir::itemlikevisit::ItemLikeVisitor;
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
            stable: FxHashMap(),
            unstable: FxHashSet(),
        }
    }

    pub fn iter(&self) -> Vec<(Symbol, Option<Symbol>)> {
        self.stable.iter().map(|(f, s)| (*f, Some(*s)))
            .chain(self.unstable.iter().map(|f| (*f, None)))
            .collect()
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

    fn extract(&self, attrs: &[Attribute]) -> Vec<(Symbol, Option<Symbol>, Span)> {
        let stab_attrs = vec!["stable", "unstable", "rustc_const_unstable"];
        let mut features = vec![];

        for attr in attrs {
            // FIXME(varkor): the stability attribute might be behind a `#[cfg]` attribute.

            // Find a stability attribute (i.e. `#[stable (..)]`, `#[unstable (..)]`,
            // `#[rustc_const_unstable (..)]`).
            if stab_attrs.iter().any(|stab_attr| attr.check_name(stab_attr)) {
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
                        features.push((feature, since, attr.span));
                    }
                    // We need to iterate over the other attributes, because
                    // `rustc_const_unstable` is not mutually exclusive with
                    // the other stability attributes, so we can't just `break`
                    // here.
                }
            }
        }

        features
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
                    if since.is_some() { "stable"} else { "unstable" },
                    if since.is_none() { "stable"} else { "unstable" },
                );
                self.tcx.sess.struct_span_err_with_code(span, &msg,
                    DiagnosticId::Error("E0711".into())).emit();
            }
        }
    }

    fn collect_from_attrs(&mut self, attrs: &[Attribute]) {
        for (feature, stable, span) in self.extract(attrs) {
            self.collect_feature(feature, stable, span);
        }
    }
}

impl<'a, 'v, 'tcx> ItemLikeVisitor<'v> for LibFeatureCollector<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        self.collect_from_attrs(&item.attrs);
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem) {
        self.collect_from_attrs(&trait_item.attrs);
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem) {
        self.collect_from_attrs(&impl_item.attrs);
    }
}

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> LibFeatures {
    let mut collector = LibFeatureCollector::new(tcx);
    for &cnum in tcx.crates().iter() {
        for &(feature, since) in tcx.defined_lib_features(cnum).iter() {
            collector.collect_feature(feature, since, DUMMY_SP);
        }
    }
    collector.collect_from_attrs(&tcx.hir.krate().attrs);
    tcx.hir.krate().visit_all_item_likes(&mut collector);
    collector.lib_features
}
