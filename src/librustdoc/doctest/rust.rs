//! Doctest functionality used only for doctests in `.rs` source files.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, intravisit};
use rustc_middle::hir::map::Map;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_resolve::rustdoc::span_of_fragments;
use rustc_session::Session;
use rustc_span::{Span, DUMMY_SP};

use super::Collector;
use crate::clean::{types::AttributesExt, Attributes};
use crate::html::markdown::{self, ErrorCodes};

pub(super) struct HirCollector<'a, 'hir, 'tcx> {
    pub(super) sess: &'a Session,
    pub(super) collector: &'a mut Collector,
    pub(super) map: Map<'hir>,
    pub(super) codes: ErrorCodes,
    pub(super) tcx: TyCtxt<'tcx>,
}

impl<'a, 'hir, 'tcx> HirCollector<'a, 'hir, 'tcx> {
    pub(super) fn visit_testable<F: FnOnce(&mut Self)>(
        &mut self,
        name: String,
        def_id: LocalDefId,
        sp: Span,
        nested: F,
    ) {
        let ast_attrs = self.tcx.hir().attrs(self.tcx.local_def_id_to_hir_id(def_id));
        if let Some(ref cfg) = ast_attrs.cfg(self.tcx, &FxHashSet::default()) {
            if !cfg.matches(&self.sess.psess, Some(self.tcx.features())) {
                return;
            }
        }

        let has_name = !name.is_empty();
        if has_name {
            self.collector.names.push(name);
        }

        // The collapse-docs pass won't combine sugared/raw doc attributes, or included files with
        // anything else, this will combine them for us.
        let attrs = Attributes::from_ast(ast_attrs);
        if let Some(doc) = attrs.opt_doc_value() {
            // Use the outermost invocation, so that doctest names come from where the docs were written.
            let span = ast_attrs
                .iter()
                .find(|attr| attr.doc_str().is_some())
                .map(|attr| attr.span.ctxt().outer_expn().expansion_cause().unwrap_or(attr.span))
                .unwrap_or(DUMMY_SP);
            self.collector.set_position(span);
            markdown::find_testable_code(
                &doc,
                self.collector,
                self.codes,
                self.collector.enable_per_target_ignores,
                Some(&crate::html::markdown::ExtraInfo::new(
                    self.tcx,
                    def_id.to_def_id(),
                    span_of_fragments(&attrs.doc_strings).unwrap_or(sp),
                )),
            );
        }

        nested(self);

        if has_name {
            self.collector.names.pop();
        }
    }
}

impl<'a, 'hir, 'tcx> intravisit::Visitor<'hir> for HirCollector<'a, 'hir, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.map
    }

    fn visit_item(&mut self, item: &'hir hir::Item<'_>) {
        let name = match &item.kind {
            hir::ItemKind::Impl(impl_) => {
                rustc_hir_pretty::id_to_string(&self.map, impl_.self_ty.hir_id)
            }
            _ => item.ident.to_string(),
        };

        self.visit_testable(name, item.owner_id.def_id, item.span, |this| {
            intravisit::walk_item(this, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'hir hir::TraitItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_trait_item(this, item);
        });
    }

    fn visit_impl_item(&mut self, item: &'hir hir::ImplItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_impl_item(this, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'hir hir::ForeignItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_foreign_item(this, item);
        });
    }

    fn visit_variant(&mut self, v: &'hir hir::Variant<'_>) {
        self.visit_testable(v.ident.to_string(), v.def_id, v.span, |this| {
            intravisit::walk_variant(this, v);
        });
    }

    fn visit_field_def(&mut self, f: &'hir hir::FieldDef<'_>) {
        self.visit_testable(f.ident.to_string(), f.def_id, f.span, |this| {
            intravisit::walk_field_def(this, f);
        });
    }
}
