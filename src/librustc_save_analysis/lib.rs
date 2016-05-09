// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_save_analysis"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use] extern crate rustc;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate serialize as rustc_serialize;

mod csv_dumper;
mod json_dumper;
mod data;
mod dump;
mod dump_visitor;
pub mod external_data;
#[macro_use]
pub mod span_utils;

use rustc::hir;
use rustc::hir::map::NodeItem;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::session::config::CrateType::CrateTypeExecutable;
use rustc::ty::{self, TyCtxt};

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use syntax::ast::{self, NodeId, PatKind};
use syntax::codemap::*;
use syntax::parse::token::{self, keywords};
use syntax::visit::{self, Visitor};
use syntax::print::pprust::ty_to_string;

pub use self::csv_dumper::CsvDumper;
pub use self::json_dumper::JsonDumper;
pub use self::data::*;
pub use self::dump::Dump;
pub use self::dump_visitor::DumpVisitor;
use self::span_utils::SpanUtils;

// FIXME this is legacy code and should be removed
pub mod recorder {
    pub use self::Row::*;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Row {
        TypeRef,
        ModRef,
        VarRef,
        FnRef,
    }
}

pub struct SaveContext<'l, 'tcx: 'l> {
    tcx: &'l TyCtxt<'tcx>,
    span_utils: SpanUtils<'tcx>,
}

macro_rules! option_try(
    ($e:expr) => (match $e { Some(e) => e, None => return None })
);

impl<'l, 'tcx: 'l> SaveContext<'l, 'tcx> {
    pub fn new(tcx: &'l TyCtxt<'tcx>) -> SaveContext<'l, 'tcx> {
        let span_utils = SpanUtils::new(&tcx.sess);
        SaveContext::from_span_utils(tcx, span_utils)
    }

    pub fn from_span_utils(tcx: &'l TyCtxt<'tcx>,
                           span_utils: SpanUtils<'tcx>)
                           -> SaveContext<'l, 'tcx> {
        SaveContext {
            tcx: tcx,
            span_utils: span_utils,
        }
    }

    // List external crates used by the current crate.
    pub fn get_external_crates(&self) -> Vec<CrateData> {
        let mut result = Vec::new();

        for n in self.tcx.sess.cstore.crates() {
            let span = match self.tcx.sess.cstore.extern_crate(n) {
                Some(ref c) => c.span,
                None => {
                    debug!("Skipping crate {}, no data", n);
                    continue;
                }
            };
            result.push(CrateData {
                name: (&self.tcx.sess.cstore.crate_name(n)[..]).to_owned(),
                number: n,
                span: span,
            });
        }

        result
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Option<Data> {
        match item.node {
            ast::ItemKind::Fn(..) => {
                let name = self.tcx.node_path_str(item.id);
                let qualname = format!("::{}", name);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Fn);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::FunctionData(FunctionData {
                    id: item.id,
                    name: name,
                    qualname: qualname,
                    declaration: None,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                }))
            }
            ast::ItemKind::Static(ref typ, mt, ref expr) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));

                // If the variable is immutable, save the initialising expression.
                let (value, keyword) = match mt {
                    ast::Mutability::Mutable => (String::from("<mutable>"), keywords::Mut),
                    ast::Mutability::Immutable => {
                        (self.span_utils.snippet(expr.span), keywords::Static)
                    },
                };

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keyword);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::VariableData(VariableData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    value: value,
                    type_value: ty_to_string(&typ),
                }))
            }
            ast::ItemKind::Const(ref typ, ref expr) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Const);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::VariableData(VariableData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    value: self.span_utils.snippet(expr.span),
                    type_value: ty_to_string(&typ),
                }))
            }
            ast::ItemKind::Mod(ref m) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));

                let cm = self.tcx.sess.codemap();
                let filename = cm.span_to_filename(m.inner);

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Mod);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::ModData(ModData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    filename: filename,
                }))
            }
            ast::ItemKind::Enum(..) => {
                let enum_name = format!("::{}", self.tcx.node_path_str(item.id));
                let val = self.span_utils.snippet(item.span);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Enum);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::EnumData(EnumData {
                    id: item.id,
                    value: val,
                    span: sub_span.unwrap(),
                    qualname: enum_name,
                    scope: self.enclosing_scope(item.id),
                }))
            }
            ast::ItemKind::Impl(_, _, _, ref trait_ref, ref typ, _) => {
                let mut type_data = None;
                let sub_span;

                let parent = self.enclosing_scope(item.id);

                match typ.node {
                    // Common case impl for a struct or something basic.
                    ast::TyKind::Path(None, ref path) => {
                        sub_span = self.span_utils.sub_span_for_type_name(path.span);
                        filter!(self.span_utils, sub_span, path.span, None);
                        type_data = self.lookup_ref_id(typ.id).map(|id| {
                            TypeRefData {
                                span: sub_span.unwrap(),
                                scope: parent,
                                ref_id: Some(id),
                                qualname: String::new() // FIXME: generate the real qualname
                            }
                        });
                    }
                    _ => {
                        // Less useful case, impl for a compound type.
                        let span = typ.span;
                        sub_span = self.span_utils.sub_span_for_type_name(span).or(Some(span));
                    }
                }

                let trait_data = trait_ref.as_ref()
                                          .and_then(|tr| self.get_trait_ref_data(tr, parent));

                filter!(self.span_utils, sub_span, typ.span, None);
                Some(Data::ImplData(ImplData2 {
                    id: item.id,
                    span: sub_span.unwrap(),
                    scope: parent,
                    trait_ref: trait_data,
                    self_ref: type_data,
                }))
            }
            _ => {
                // FIXME
                bug!();
            }
        }
    }

    pub fn get_field_data(&self, field: &ast::StructField,
                          scope: NodeId) -> Option<VariableData> {
        if let Some(ident) = field.ident {
            let qualname = format!("::{}::{}", self.tcx.node_path_str(scope), ident);
            let typ = self.tcx.node_types().get(&field.id).unwrap().to_string();
            let sub_span = self.span_utils.sub_span_before_token(field.span, token::Colon);
            filter!(self.span_utils, sub_span, field.span, None);
            Some(VariableData {
                id: field.id,
                name: ident.to_string(),
                qualname: qualname,
                span: sub_span.unwrap(),
                scope: scope,
                value: "".to_owned(),
                type_value: typ,
            })
        } else {
            None
        }
    }

    // FIXME would be nice to take a MethodItem here, but the ast provides both
    // trait and impl flavours, so the caller must do the disassembly.
    pub fn get_method_data(&self, id: ast::NodeId,
                           name: ast::Name, span: Span) -> Option<FunctionData> {
        // The qualname for a method is the trait name or name of the struct in an impl in
        // which the method is declared in, followed by the method's name.
        let qualname = match self.tcx.impl_of_method(self.tcx.map.local_def_id(id)) {
            Some(impl_id) => match self.tcx.map.get_if_local(impl_id) {
                Some(NodeItem(item)) => {
                    match item.node {
                        hir::ItemImpl(_, _, _, _, ref ty, _) => {
                            let mut result = String::from("<");
                            result.push_str(&rustc::hir::print::ty_to_string(&ty));

                            match self.tcx.trait_of_item(self.tcx.map.local_def_id(id)) {
                                Some(def_id) => {
                                    result.push_str(" as ");
                                    result.push_str(&self.tcx.item_path_str(def_id));
                                }
                                None => {}
                            }
                            result.push_str(">");
                            result
                        }
                        _ => {
                            span_bug!(span,
                                      "Container {:?} for method {} not an impl?",
                                      impl_id,
                                      id);
                        }
                    }
                }
                r => {
                    span_bug!(span,
                              "Container {:?} for method {} is not a node item {:?}",
                              impl_id,
                              id,
                              r);
                }
            },
            None => match self.tcx.trait_of_item(self.tcx.map.local_def_id(id)) {
                Some(def_id) => {
                    match self.tcx.map.get_if_local(def_id) {
                        Some(NodeItem(_)) => {
                            format!("::{}", self.tcx.item_path_str(def_id))
                        }
                        r => {
                            span_bug!(span,
                                      "Could not find container {:?} for \
                                       method {}, got {:?}",
                                      def_id,
                                      id,
                                      r);
                        }
                    }
                }
                None => {
                    span_bug!(span, "Could not find container for method {}", id);
                }
            },
        };

        let qualname = format!("{}::{}", qualname, name);

        let def_id = self.tcx.map.local_def_id(id);
        let decl_id = self.tcx.trait_item_of_item(def_id).and_then(|new_id| {
            let new_def_id = new_id.def_id();
            if new_def_id != def_id {
                Some(new_def_id)
            } else {
                None
            }
        });

        let sub_span = self.span_utils.sub_span_after_keyword(span, keywords::Fn);
        filter!(self.span_utils, sub_span, span, None);
        Some(FunctionData {
            id: id,
            name: name.to_string(),
            qualname: qualname,
            declaration: decl_id,
            span: sub_span.unwrap(),
            scope: self.enclosing_scope(id),
        })
    }

    pub fn get_trait_ref_data(&self,
                              trait_ref: &ast::TraitRef,
                              parent: NodeId)
                              -> Option<TypeRefData> {
        self.lookup_ref_id(trait_ref.ref_id).and_then(|def_id| {
            let span = trait_ref.path.span;
            let sub_span = self.span_utils.sub_span_for_type_name(span).or(Some(span));
            filter!(self.span_utils, sub_span, span, None);
            Some(TypeRefData {
                span: sub_span.unwrap(),
                scope: parent,
                ref_id: Some(def_id),
                qualname: String::new() // FIXME: generate the real qualname
            })
        })
    }

    pub fn get_expr_data(&self, expr: &ast::Expr) -> Option<Data> {
        let hir_node = self.tcx.map.expect_expr(expr.id);
        let ty = self.tcx.expr_ty_adjusted_opt(&hir_node);
        if ty.is_none() || ty.unwrap().sty == ty::TyError {
            return None;
        }
        match expr.node {
            ast::ExprKind::Field(ref sub_ex, ident) => {
                let hir_node = self.tcx.map.expect_expr(sub_ex.id);
                match self.tcx.expr_ty_adjusted(&hir_node).sty {
                    ty::TyStruct(def, _) => {
                        let f = def.struct_variant().field_named(ident.node.name);
                        let sub_span = self.span_utils.span_for_last_ident(expr.span);
                        filter!(self.span_utils, sub_span, expr.span, None);
                        return Some(Data::VariableRefData(VariableRefData {
                            name: ident.node.to_string(),
                            span: sub_span.unwrap(),
                            scope: self.enclosing_scope(expr.id),
                            ref_id: f.did,
                        }));
                    }
                    _ => {
                        debug!("Expected struct type, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprKind::Struct(ref path, _, _) => {
                let hir_node = self.tcx.map.expect_expr(expr.id);
                match self.tcx.expr_ty_adjusted(&hir_node).sty {
                    ty::TyStruct(def, _) => {
                        let sub_span = self.span_utils.span_for_last_ident(path.span);
                        filter!(self.span_utils, sub_span, path.span, None);
                        Some(Data::TypeRefData(TypeRefData {
                            span: sub_span.unwrap(),
                            scope: self.enclosing_scope(expr.id),
                            ref_id: Some(def.did),
                            qualname: String::new() // FIXME: generate the real qualname
                        }))
                    }
                    _ => {
                        // FIXME ty could legitimately be a TyEnum, but then we will fail
                        // later if we try to look up the fields.
                        debug!("expected TyStruct, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprKind::MethodCall(..) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method_id = self.tcx.tables.borrow().method_map[&method_call].def_id;
                let (def_id, decl_id) = match self.tcx.impl_or_trait_item(method_id).container() {
                    ty::ImplContainer(_) => (Some(method_id), None),
                    ty::TraitContainer(_) => (None, Some(method_id)),
                };
                let sub_span = self.span_utils.sub_span_for_meth_name(expr.span);
                filter!(self.span_utils, sub_span, expr.span, None);
                let parent = self.enclosing_scope(expr.id);
                Some(Data::MethodCallData(MethodCallData {
                    span: sub_span.unwrap(),
                    scope: parent,
                    ref_id: def_id,
                    decl_id: decl_id,
                }))
            }
            ast::ExprKind::Path(_, ref path) => {
                self.get_path_data(expr.id, path)
            }
            _ => {
                // FIXME
                bug!();
            }
        }
    }

    pub fn get_path_data(&self, id: NodeId, path: &ast::Path) -> Option<Data> {
        let def_map = self.tcx.def_map.borrow();
        if !def_map.contains_key(&id) {
            span_bug!(path.span, "def_map has no key for {} in visit_expr", id);
        }
        let def = def_map.get(&id).unwrap().full_def();
        let sub_span = self.span_utils.span_for_last_ident(path.span);
        filter!(self.span_utils, sub_span, path.span, None);
        match def {
            Def::Upvar(..) |
            Def::Local(..) |
            Def::Static(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Variant(..) => {
                Some(Data::VariableRefData(VariableRefData {
                    name: self.span_utils.snippet(sub_span.unwrap()),
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                    ref_id: def.def_id(),
                }))
            }
            Def::Struct(def_id) |
            Def::Enum(def_id) |
            Def::TyAlias(def_id) |
            Def::Trait(def_id) |
            Def::TyParam(_, _, def_id, _) => {
                Some(Data::TypeRefData(TypeRefData {
                    span: sub_span.unwrap(),
                    ref_id: Some(def_id),
                    scope: self.enclosing_scope(id),
                    qualname: String::new() // FIXME: generate the real qualname
                }))
            }
            Def::Method(decl_id) => {
                let sub_span = self.span_utils.sub_span_for_meth_name(path.span);
                filter!(self.span_utils, sub_span, path.span, None);
                let def_id = if decl_id.is_local() {
                    let ti = self.tcx.impl_or_trait_item(decl_id);
                    match ti.container() {
                        ty::TraitContainer(def_id) => {
                            self.tcx
                                .trait_items(def_id)
                                .iter()
                                .find(|mr| mr.name() == ti.name() && self.trait_method_has_body(mr))
                                .map(|mr| mr.def_id())
                        }
                        ty::ImplContainer(def_id) => {
                            let impl_items = self.tcx.impl_items.borrow();
                            Some(impl_items.get(&def_id)
                                           .unwrap()
                                           .iter()
                                           .find(|mr| {
                                               self.tcx.impl_or_trait_item(mr.def_id()).name() ==
                                               ti.name()
                                           })
                                           .unwrap()
                                           .def_id())
                        }
                    }
                } else {
                    None
                };
                Some(Data::MethodCallData(MethodCallData {
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                    ref_id: def_id,
                    decl_id: Some(decl_id),
                }))
            }
            Def::Fn(def_id) => {
                Some(Data::FunctionCallData(FunctionCallData {
                    ref_id: def_id,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                }))
            }
            Def::Mod(def_id) => {
                Some(Data::ModRefData(ModRefData {
                    ref_id: Some(def_id),
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                    qualname: String::new() // FIXME: generate the real qualname
                }))
            }
            _ => None,
        }
    }

    fn trait_method_has_body(&self, mr: &ty::ImplOrTraitItem) -> bool {
        let def_id = mr.def_id();
        if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
            let trait_item = self.tcx.map.expect_trait_item(node_id);
            if let hir::TraitItem_::MethodTraitItem(_, Some(_)) = trait_item.node {
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn get_field_ref_data(&self,
                              field_ref: &ast::Field,
                              variant: ty::VariantDef,
                              parent: NodeId)
                              -> Option<VariableRefData> {
        let f = variant.field_named(field_ref.ident.node.name);
        // We don't really need a sub-span here, but no harm done
        let sub_span = self.span_utils.span_for_last_ident(field_ref.ident.span);
        filter!(self.span_utils, sub_span, field_ref.ident.span, None);
        Some(VariableRefData {
            name: field_ref.ident.node.to_string(),
            span: sub_span.unwrap(),
            scope: parent,
            ref_id: f.did,
        })
    }

    /// Attempt to return MacroUseData for any AST node.
    ///
    /// For a given piece of AST defined by the supplied Span and NodeId,
    /// returns None if the node is not macro-generated or the span is malformed,
    /// else uses the expansion callsite and callee to return some MacroUseData.
    pub fn get_macro_use_data(&self, span: Span, id: NodeId) -> Option<MacroUseData> {
        if !generated_code(span) {
            return None;
        }
        // Note we take care to use the source callsite/callee, to handle
        // nested expansions and ensure we only generate data for source-visible
        // macro uses.
        let callsite = self.tcx.sess.codemap().source_callsite(span);
        let callee = self.tcx.sess.codemap().source_callee(span);
        let callee = option_try!(callee);
        let callee_span = option_try!(callee.span);

        // Ignore attribute macros, their spans are usually mangled
        if let MacroAttribute(_) = callee.format {
            return None;
        }

        // If the callee is an imported macro from an external crate, need to get
        // the source span and name from the session, as their spans are localized
        // when read in, and no longer correspond to the source.
        if let Some(mac) = self.tcx.sess.imported_macro_spans.borrow().get(&callee_span) {
            let &(ref mac_name, mac_span) = mac;
            return Some(MacroUseData {
                                        span: callsite,
                                        name: mac_name.clone(),
                                        callee_span: mac_span,
                                        scope: self.enclosing_scope(id),
                                        imported: true,
                                        qualname: String::new()// FIXME: generate the real qualname
                                    });
        }

        Some(MacroUseData {
            span: callsite,
            name: callee.name().to_string(),
            callee_span: callee_span,
            scope: self.enclosing_scope(id),
            imported: false,
            qualname: String::new() // FIXME: generate the real qualname
        })
    }

    pub fn get_data_for_id(&self, _id: &NodeId) -> Data {
        // FIXME
        bug!();
    }

    fn lookup_ref_id(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.tcx.def_map.borrow().contains_key(&ref_id) {
            bug!("def_map has no key for {} in lookup_type_ref", ref_id);
        }
        let def = self.tcx.def_map.borrow().get(&ref_id).unwrap().full_def();
        match def {
            Def::PrimTy(_) | Def::SelfTy(..) => None,
            _ => Some(def.def_id()),
        }
    }

    #[inline]
    pub fn enclosing_scope(&self, id: NodeId) -> NodeId {
        self.tcx.map.get_enclosing_scope(id).unwrap_or(0)
    }
}

// An AST visitor for collecting paths from patterns.
struct PathCollector {
    // The Row field identifies the kind of pattern.
    collected_paths: Vec<(NodeId, ast::Path, ast::Mutability, recorder::Row)>,
}

impl PathCollector {
    fn new() -> PathCollector {
        PathCollector { collected_paths: vec![] }
    }
}

impl<'v> Visitor<'v> for PathCollector {
    fn visit_pat(&mut self, p: &ast::Pat) {
        match p.node {
            PatKind::Struct(ref path, _, _) => {
                self.collected_paths.push((p.id, path.clone(),
                                           ast::Mutability::Mutable, recorder::TypeRef));
            }
            PatKind::TupleStruct(ref path, _) |
            PatKind::Path(ref path) |
            PatKind::QPath(_, ref path) => {
                self.collected_paths.push((p.id, path.clone(),
                                           ast::Mutability::Mutable, recorder::VarRef));
            }
            PatKind::Ident(bm, ref path1, _) => {
                debug!("PathCollector, visit ident in pat {}: {:?} {:?}",
                       path1.node,
                       p.span,
                       path1.span);
                let immut = match bm {
                    // Even if the ref is mut, you can't change the ref, only
                    // the data pointed at, so showing the initialising expression
                    // is still worthwhile.
                    ast::BindingMode::ByRef(_) => ast::Mutability::Immutable,
                    ast::BindingMode::ByValue(mt) => mt,
                };
                // collect path for either visit_local or visit_arm
                let path = ast::Path::from_ident(path1.span, path1.node);
                self.collected_paths.push((p.id, path, immut, recorder::VarRef));
            }
            _ => {}
        }
        visit::walk_pat(self, p);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Csv,
    Json,
}

impl Format {
    fn extension(&self) -> &'static str {
        match *self {
            Format::Csv => ".csv",
            Format::Json => ".json",
        }
    }
}

pub fn process_crate<'l, 'tcx>(tcx: &'l TyCtxt<'tcx>,
                               krate: &ast::Crate,
                               analysis: &'l ty::CrateAnalysis<'l>,
                               cratename: &str,
                               odir: Option<&Path>,
                               format: Format) {
    let _ignore = tcx.dep_graph.in_ignore();

    assert!(analysis.glob_map.is_some());

    info!("Dumping crate {}", cratename);

    // find a path to dump our data to
    let mut root_path = match env::var_os("RUST_SAVE_ANALYSIS_FOLDER") {
        Some(val) => PathBuf::from(val),
        None => match odir {
            Some(val) => val.join("save-analysis"),
            None => PathBuf::from("save-analysis-temp"),
        },
    };

    if let Err(e) = fs::create_dir_all(&root_path) {
        tcx.sess.err(&format!("Could not create directory {}: {}",
                              root_path.display(),
                              e));
    }

    {
        let disp = root_path.display();
        info!("Writing output to {}", disp);
    }

    // Create output file.
    let executable = tcx.sess.crate_types.borrow().iter().any(|ct| *ct == CrateTypeExecutable);
    let mut out_name = if executable {
        "".to_owned()
    } else {
        "lib".to_owned()
    };
    out_name.push_str(&cratename);
    out_name.push_str(&tcx.sess.opts.cg.extra_filename);
    out_name.push_str(format.extension());
    root_path.push(&out_name);
    let mut output_file = File::create(&root_path).unwrap_or_else(|e| {
        let disp = root_path.display();
        tcx.sess.fatal(&format!("Could not open {}: {}", disp, e));
    });
    root_path.pop();
    let output = &mut output_file;

    let save_ctxt = SaveContext::new(tcx);

    macro_rules! dump {
        ($new_dumper: expr) => {{
            let mut dumper = $new_dumper;
            let mut visitor = DumpVisitor::new(tcx, save_ctxt, analysis, &mut dumper);

            visitor.dump_crate_info(cratename, krate);
            visit::walk_crate(&mut visitor, krate);
        }}
    }

    match format {
        Format::Csv => dump!(CsvDumper::new(output)),
        Format::Json => dump!(JsonDumper::new(output)),
    }
}

// Utility functions for the module.

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

// Helper function to determine if a span came from a
// macro expansion or syntax extension.
pub fn generated_code(span: Span) -> bool {
    span.expn_id != NO_EXPANSION || span == DUMMY_SP
}
