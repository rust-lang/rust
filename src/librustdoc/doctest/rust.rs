//! Doctest functionality used only for doctests in `.rs` source files.

use std::env;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_hir::{self as hir, CRATE_HIR_ID, intravisit};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_resolve::rustdoc::span_of_fragments;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, DUMMY_SP, FileName, Pos, Span};

use super::{DocTestVisitor, ScrapedDocTest};
use crate::clean::Attributes;
use crate::clean::types::AttributesExt;
use crate::html::markdown::{self, ErrorCodes, LangString, MdRelLine};

struct RustCollector {
    source_map: Lrc<SourceMap>,
    tests: Vec<ScrapedDocTest>,
    cur_path: Vec<String>,
    position: Span,
}

impl RustCollector {
    fn get_filename(&self) -> FileName {
        let filename = self.source_map.span_to_filename(self.position);
        if let FileName::Real(ref filename) = filename {
            let path = filename.remapped_path_if_available();
            // Strip the cwd prefix from the path. This will likely exist if
            // the path was not remapped.
            let path = env::current_dir()
                .map(|cur_dir| path.strip_prefix(&cur_dir).unwrap_or(path))
                .unwrap_or(path);
            return path.to_owned().into();
        }
        filename
    }

    fn get_base_line(&self) -> usize {
        let sp_lo = self.position.lo().to_usize();
        let loc = self.source_map.lookup_char_pos(BytePos(sp_lo as u32));
        loc.line
    }
}

impl DocTestVisitor for RustCollector {
    fn visit_test(&mut self, test: String, config: LangString, rel_line: MdRelLine) {
        let line = self.get_base_line() + rel_line.offset();
        self.tests.push(ScrapedDocTest::new(
            self.get_filename(),
            line,
            self.cur_path.clone(),
            config,
            test,
        ));
    }

    fn visit_header(&mut self, _name: &str, _level: u32) {}
}

pub(super) struct HirCollector<'tcx> {
    codes: ErrorCodes,
    tcx: TyCtxt<'tcx>,
    enable_per_target_ignores: bool,
    collector: RustCollector,
}

impl<'tcx> HirCollector<'tcx> {
    pub fn new(codes: ErrorCodes, enable_per_target_ignores: bool, tcx: TyCtxt<'tcx>) -> Self {
        let collector = RustCollector {
            source_map: tcx.sess.psess.clone_source_map(),
            cur_path: vec![],
            position: DUMMY_SP,
            tests: vec![],
        };
        Self { codes, enable_per_target_ignores, tcx, collector }
    }

    pub fn collect_crate(mut self) -> Vec<ScrapedDocTest> {
        let tcx = self.tcx;
        self.visit_testable("".to_string(), CRATE_DEF_ID, tcx.hir().span(CRATE_HIR_ID), |this| {
            tcx.hir().walk_toplevel_module(this)
        });
        self.collector.tests
    }
}

impl HirCollector<'_> {
    fn visit_testable<F: FnOnce(&mut Self)>(
        &mut self,
        name: String,
        def_id: LocalDefId,
        sp: Span,
        nested: F,
    ) {
        let ast_attrs = self.tcx.hir().attrs(self.tcx.local_def_id_to_hir_id(def_id));
        if let Some(ref cfg) = ast_attrs.cfg(self.tcx, &FxHashSet::default()) {
            if !cfg.matches(&self.tcx.sess.psess, Some(self.tcx.features())) {
                return;
            }
        }

        let has_name = !name.is_empty();
        if has_name {
            self.collector.cur_path.push(name);
        }

        // The collapse-docs pass won't combine sugared/raw doc attributes, or included files with
        // anything else, this will combine them for us.
        let attrs = Attributes::from_hir(ast_attrs);
        if let Some(doc) = attrs.opt_doc_value() {
            let span = span_of_fragments(&attrs.doc_strings).unwrap_or(sp);
            self.collector.position = if span.edition().at_least_rust_2024() {
                span
            } else {
                // this span affects filesystem path resolution,
                // so we need to keep it the same as it was previously
                ast_attrs
                    .iter()
                    .find(|attr| attr.doc_str().is_some())
                    .map(|attr| {
                        attr.span.ctxt().outer_expn().expansion_cause().unwrap_or(attr.span)
                    })
                    .unwrap_or(DUMMY_SP)
            };
            markdown::find_testable_code(
                &doc,
                &mut self.collector,
                self.codes,
                self.enable_per_target_ignores,
                Some(&crate::html::markdown::ExtraInfo::new(self.tcx, def_id, span)),
            );
        }

        nested(self);

        if has_name {
            self.collector.cur_path.pop();
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for HirCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'_>) {
        let name = match &item.kind {
            hir::ItemKind::Impl(impl_) => {
                rustc_hir_pretty::id_to_string(&self.tcx.hir(), impl_.self_ty.hir_id)
            }
            _ => item.ident.to_string(),
        };

        self.visit_testable(name, item.owner_id.def_id, item.span, |this| {
            intravisit::walk_item(this, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_trait_item(this, item);
        });
    }

    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_impl_item(this, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.owner_id.def_id, item.span, |this| {
            intravisit::walk_foreign_item(this, item);
        });
    }

    fn visit_variant(&mut self, v: &'tcx hir::Variant<'_>) {
        self.visit_testable(v.ident.to_string(), v.def_id, v.span, |this| {
            intravisit::walk_variant(this, v);
        });
    }

    fn visit_field_def(&mut self, f: &'tcx hir::FieldDef<'_>) {
        self.visit_testable(f.ident.to_string(), f.def_id, f.span, |this| {
            intravisit::walk_field_def(this, f);
        });
    }
}
