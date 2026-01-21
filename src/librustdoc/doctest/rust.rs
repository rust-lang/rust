//! Doctest functionality used only for doctests in `.rs` source files.

use std::cell::Cell;
use std::str::FromStr;
use std::sync::Arc;

use proc_macro2::{TokenStream, TokenTree};
use rustc_attr_parsing::eval_config_entry;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_hir::{self as hir, Attribute, CRATE_HIR_ID, intravisit};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_resolve::rustdoc::span_of_fragments;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, DUMMY_SP, FileName, Pos, Span};

use super::{DocTestVisitor, ScrapedDocTest};
use crate::clean::{Attributes, CfgInfo, extract_cfg_from_attrs};
use crate::html::markdown::{self, ErrorCodes, LangString, MdRelLine};

struct RustCollector {
    source_map: Arc<SourceMap>,
    tests: Vec<ScrapedDocTest>,
    cur_path: Vec<String>,
    position: Span,
    global_crate_attrs: Vec<String>,
}

impl RustCollector {
    fn get_filename(&self) -> FileName {
        let filename = self.source_map.span_to_filename(self.position);
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
        let base_line = self.get_base_line();
        let line = base_line + rel_line.offset();
        let count = Cell::new(base_line);
        let span = if line > base_line {
            match self.source_map.span_extend_while(self.position, |c| {
                if c == '\n' {
                    let count_v = count.get();
                    count.set(count_v + 1);
                    if count_v >= line {
                        return false;
                    }
                }
                true
            }) {
                Ok(sp) => self.source_map.span_extend_to_line(sp.shrink_to_hi()),
                _ => self.position,
            }
        } else {
            self.position
        };
        self.tests.push(ScrapedDocTest::new(
            self.get_filename(),
            line,
            self.cur_path.clone(),
            config,
            test,
            span,
            self.global_crate_attrs.clone(),
        ));
    }

    fn visit_header(&mut self, _name: &str, _level: u32) {}
}

pub(super) struct HirCollector<'tcx> {
    codes: ErrorCodes,
    tcx: TyCtxt<'tcx>,
    collector: RustCollector,
}

impl<'tcx> HirCollector<'tcx> {
    pub fn new(codes: ErrorCodes, tcx: TyCtxt<'tcx>) -> Self {
        let collector = RustCollector {
            source_map: tcx.sess.psess.clone_source_map(),
            cur_path: vec![],
            position: DUMMY_SP,
            tests: vec![],
            global_crate_attrs: Vec::new(),
        };
        Self { codes, tcx, collector }
    }

    pub fn collect_crate(mut self) -> Vec<ScrapedDocTest> {
        let tcx = self.tcx;
        self.visit_testable(None, CRATE_DEF_ID, tcx.hir_span(CRATE_HIR_ID), |this| {
            tcx.hir_walk_toplevel_module(this)
        });
        self.collector.tests
    }
}

impl HirCollector<'_> {
    fn visit_testable<F: FnOnce(&mut Self)>(
        &mut self,
        name: Option<String>,
        def_id: LocalDefId,
        sp: Span,
        nested: F,
    ) {
        let ast_attrs = self.tcx.hir_attrs(self.tcx.local_def_id_to_hir_id(def_id));
        if let Some(ref cfg) =
            extract_cfg_from_attrs(ast_attrs.iter(), self.tcx, &mut CfgInfo::default())
            && !eval_config_entry(&self.tcx.sess, cfg.inner()).as_bool()
        {
            return;
        }

        let source_map = self.tcx.sess.source_map();
        // Try collecting `#[doc(test(attr(...)))]`
        let old_global_crate_attrs_len = self.collector.global_crate_attrs.len();
        for attr in ast_attrs {
            let Attribute::Parsed(AttributeKind::Doc(d)) = attr else { continue };
            for attr_span in &d.test_attrs {
                // FIXME: This is ugly, remove when `test_attrs` has been ported to new attribute API.
                if let Ok(snippet) = source_map.span_to_snippet(*attr_span)
                    && let Ok(stream) = TokenStream::from_str(&snippet)
                {
                    let mut iter = stream.into_iter().peekable();
                    while let Some(token) = iter.next() {
                        if let TokenTree::Ident(i) = token {
                            let i = i.to_string();
                            let peek = iter.peek();
                            // From this ident, we can have things like:
                            //
                            // * Group: `allow(...)`
                            // * Name/value: `crate_name = "..."`
                            // * Tokens: `html_no_url`
                            //
                            // So we peek next element to know what case we are in.
                            match peek {
                                Some(TokenTree::Group(g)) => {
                                    let g = g.to_string();
                                    iter.next();
                                    // Add the additional attributes to the global_crate_attrs vector
                                    self.collector.global_crate_attrs.push(format!("{i}{g}"));
                                }
                                // If next item is `=`, it means it's a name value so we will need
                                // to get the value as well.
                                Some(TokenTree::Punct(p)) if p.as_char() == '=' => {
                                    let p = p.to_string();
                                    iter.next();
                                    if let Some(last) = iter.next() {
                                        // Add the additional attributes to the global_crate_attrs vector
                                        self.collector
                                            .global_crate_attrs
                                            .push(format!("{i}{p}{last}"));
                                    }
                                }
                                _ => {
                                    // Add the additional attributes to the global_crate_attrs vector
                                    self.collector.global_crate_attrs.push(i.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut has_name = false;
        if let Some(name) = name {
            self.collector.cur_path.push(name);
            has_name = true;
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
                        attr.span().ctxt().outer_expn().expansion_cause().unwrap_or(attr.span())
                    })
                    .unwrap_or(DUMMY_SP)
            };
            markdown::find_testable_code(
                &doc,
                &mut self.collector,
                self.codes,
                Some(&crate::html::markdown::ExtraInfo::new(self.tcx, def_id, span)),
            );
        }

        nested(self);

        // Restore global_crate_attrs to it's previous size/content
        self.collector.global_crate_attrs.truncate(old_global_crate_attrs_len);

        if has_name {
            self.collector.cur_path.pop();
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for HirCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'_>) {
        let name = match &item.kind {
            hir::ItemKind::Impl(impl_) => {
                Some(rustc_hir_pretty::id_to_string(&self.tcx, impl_.self_ty.hir_id))
            }
            _ => item.kind.ident().map(|ident| ident.to_string()),
        };

        self.visit_testable(name, item.owner_id.def_id, item.span, |this| {
            intravisit::walk_item(this, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem<'_>) {
        self.visit_testable(
            Some(item.ident.to_string()),
            item.owner_id.def_id,
            item.span,
            |this| {
                intravisit::walk_trait_item(this, item);
            },
        );
    }

    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem<'_>) {
        self.visit_testable(
            Some(item.ident.to_string()),
            item.owner_id.def_id,
            item.span,
            |this| {
                intravisit::walk_impl_item(this, item);
            },
        );
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'_>) {
        self.visit_testable(
            Some(item.ident.to_string()),
            item.owner_id.def_id,
            item.span,
            |this| {
                intravisit::walk_foreign_item(this, item);
            },
        );
    }

    fn visit_variant(&mut self, v: &'tcx hir::Variant<'_>) {
        self.visit_testable(Some(v.ident.to_string()), v.def_id, v.span, |this| {
            intravisit::walk_variant(this, v);
        });
    }

    fn visit_field_def(&mut self, f: &'tcx hir::FieldDef<'_>) {
        self.visit_testable(Some(f.ident.to_string()), f.def_id, f.span, |this| {
            intravisit::walk_field_def(this, f);
        });
    }
}
