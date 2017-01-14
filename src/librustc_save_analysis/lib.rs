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
#![deny(warnings)]

#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use] extern crate rustc;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate serialize as rustc_serialize;
extern crate syntax_pos;


mod csv_dumper;
mod json_api_dumper;
mod json_dumper;
mod data;
mod dump;
mod dump_visitor;
pub mod external_data;
#[macro_use]
pub mod span_utils;

use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::map::Node;
use rustc::hir::def_id::DefId;
use rustc::session::config::CrateType::CrateTypeExecutable;
use rustc::ty::{self, TyCtxt};

use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};

use syntax::ast::{self, NodeId, PatKind, Attribute, CRATE_NODE_ID};
use syntax::parse::lexer::comments::strip_doc_comment_decoration;
use syntax::parse::token;
use syntax::symbol::{Symbol, keywords};
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{ty_to_string, arg_to_string};
use syntax::codemap::MacroAttribute;
use syntax_pos::*;

pub use self::csv_dumper::CsvDumper;
pub use self::json_api_dumper::JsonApiDumper;
pub use self::json_dumper::JsonDumper;
pub use self::data::*;
pub use self::external_data::make_def_id;
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
    tcx: TyCtxt<'l, 'tcx, 'tcx>,
    tables: &'l ty::Tables<'tcx>,
    analysis: &'l ty::CrateAnalysis<'tcx>,
    span_utils: SpanUtils<'tcx>,
}

macro_rules! option_try(
    ($e:expr) => (match $e { Some(e) => e, None => return None })
);

impl<'l, 'tcx: 'l> SaveContext<'l, 'tcx> {
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
                name: self.tcx.sess.cstore.crate_name(n).to_string(),
                number: n.as_u32(),
                span: span,
            });
        }

        result
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Option<Data> {
        match item.node {
            ast::ItemKind::Fn(ref decl, .., ref generics, _) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Fn);
                filter!(self.span_utils, sub_span, item.span, None);


                Some(Data::FunctionData(FunctionData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    declaration: None,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    value: make_signature(decl, generics),
                    visibility: From::from(&item.vis),
                    parent: None,
                    docs: docs_for_attrs(&item.attrs),
                    sig: self.sig_base(item),
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
                    kind: VariableKind::Static,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    parent: None,
                    value: value,
                    type_value: ty_to_string(&typ),
                    visibility: From::from(&item.vis),
                    docs: docs_for_attrs(&item.attrs),
                    sig: Some(self.sig_base(item)),
                }))
            }
            ast::ItemKind::Const(ref typ, ref expr) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Const);
                filter!(self.span_utils, sub_span, item.span, None);
                Some(Data::VariableData(VariableData {
                    id: item.id,
                    kind: VariableKind::Const,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    parent: None,
                    value: self.span_utils.snippet(expr.span),
                    type_value: ty_to_string(&typ),
                    visibility: From::from(&item.vis),
                    docs: docs_for_attrs(&item.attrs),
                    sig: Some(self.sig_base(item)),
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
                    items: m.items.iter().map(|i| i.id).collect(),
                    visibility: From::from(&item.vis),
                    docs: docs_for_attrs(&item.attrs),
                    sig: self.sig_base(item),
                }))
            }
            ast::ItemKind::Enum(ref def, _) => {
                let name = item.ident.to_string();
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Enum);
                filter!(self.span_utils, sub_span, item.span, None);
                let variants_str = def.variants.iter()
                                      .map(|v| v.node.name.to_string())
                                      .collect::<Vec<_>>()
                                      .join(", ");
                let val = format!("{}::{{{}}}", name, variants_str);
                Some(Data::EnumData(EnumData {
                    id: item.id,
                    name: name,
                    value: val,
                    span: sub_span.unwrap(),
                    qualname: qualname,
                    scope: self.enclosing_scope(item.id),
                    variants: def.variants.iter().map(|v| v.node.data.id()).collect(),
                    visibility: From::from(&item.vis),
                    docs: docs_for_attrs(&item.attrs),
                    sig: self.sig_base(item),
                }))
            }
            ast::ItemKind::Impl(.., ref trait_ref, ref typ, _) => {
                let mut type_data = None;
                let sub_span;

                let parent = self.enclosing_scope(item.id);

                match typ.node {
                    // Common case impl for a struct or something basic.
                    ast::TyKind::Path(None, ref path) => {
                        filter!(self.span_utils, None, path.span, None);
                        sub_span = self.span_utils.sub_span_for_type_name(path.span);
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

    pub fn get_field_data(&self,
                          field: &ast::StructField,
                          scope: NodeId)
                          -> Option<VariableData> {
        if let Some(ident) = field.ident {
            let name = ident.to_string();
            let qualname = format!("::{}::{}", self.tcx.node_path_str(scope), ident);
            let sub_span = self.span_utils.sub_span_before_token(field.span, token::Colon);
            filter!(self.span_utils, sub_span, field.span, None);
            let def_id = self.tcx.map.local_def_id(field.id);
            let typ = self.tcx.item_type(def_id).to_string();

            let span = field.span;
            let text = self.span_utils.snippet(field.span);
            let ident_start = text.find(&name).unwrap();
            let ident_end = ident_start + name.len();
            let sig = Signature {
                span: span,
                text: text,
                ident_start: ident_start,
                ident_end: ident_end,
                defs: vec![],
                refs: vec![],
            };
            Some(VariableData {
                id: field.id,
                kind: VariableKind::Field,
                name: name,
                qualname: qualname,
                span: sub_span.unwrap(),
                scope: scope,
                parent: Some(make_def_id(scope, &self.tcx.map)),
                value: "".to_owned(),
                type_value: typ,
                visibility: From::from(&field.vis),
                docs: docs_for_attrs(&field.attrs),
                sig: Some(sig),
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
        let (qualname, parent_scope, decl_id, vis, docs) =
          match self.tcx.impl_of_method(self.tcx.map.local_def_id(id)) {
            Some(impl_id) => match self.tcx.map.get_if_local(impl_id) {
                Some(Node::NodeItem(item)) => {
                    match item.node {
                        hir::ItemImpl(.., ref ty, _) => {
                            let mut result = String::from("<");
                            result.push_str(&self.tcx.map.node_to_pretty_string(ty.id));

                            let trait_id = self.tcx.trait_id_of_impl(impl_id);
                            let mut decl_id = None;
                            if let Some(def_id) = trait_id {
                                result.push_str(" as ");
                                result.push_str(&self.tcx.item_path_str(def_id));
                                self.tcx.associated_items(def_id)
                                    .find(|item| item.name == name)
                                    .map(|item| decl_id = Some(item.def_id));
                            }
                            result.push_str(">");

                            (result, trait_id, decl_id,
                             From::from(&item.vis),
                             docs_for_attrs(&item.attrs))
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
                        Some(Node::NodeItem(item)) => {
                            (format!("::{}", self.tcx.item_path_str(def_id)),
                             Some(def_id), None,
                             From::from(&item.vis),
                             docs_for_attrs(&item.attrs))
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

        let sub_span = self.span_utils.sub_span_after_keyword(span, keywords::Fn);
        filter!(self.span_utils, sub_span, span, None);

        let name = name.to_string();
        let text = self.span_utils.signature_string_for_span(span);
        let ident_start = text.find(&name).unwrap();
        let ident_end = ident_start + name.len();
        let sig = Signature {
            span: span,
            text: text,
            ident_start: ident_start,
            ident_end: ident_end,
            defs: vec![],
            refs: vec![],
        };

        Some(FunctionData {
            id: id,
            name: name,
            qualname: qualname,
            declaration: decl_id,
            span: sub_span.unwrap(),
            scope: self.enclosing_scope(id),
            // FIXME you get better data here by using the visitor.
            value: String::new(),
            visibility: vis,
            parent: parent_scope,
            docs: docs,
            sig: sig,
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
        let ty = self.tables.expr_ty_adjusted_opt(&hir_node);
        if ty.is_none() || ty.unwrap().sty == ty::TyError {
            return None;
        }
        match expr.node {
            ast::ExprKind::Field(ref sub_ex, ident) => {
                let hir_node = match self.tcx.map.find(sub_ex.id) {
                    Some(Node::NodeExpr(expr)) => expr,
                    _ => {
                        debug!("Missing or weird node for sub-expression {} in {:?}",
                               sub_ex.id, expr);
                        return None;
                    }
                };
                match self.tables.expr_ty_adjusted(&hir_node).sty {
                    ty::TyAdt(def, _) if !def.is_enum() => {
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
                        debug!("Expected struct or union type, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprKind::Struct(ref path, ..) => {
                match self.tables.expr_ty_adjusted(&hir_node).sty {
                    ty::TyAdt(def, _) if !def.is_enum() => {
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
                        // FIXME ty could legitimately be an enum, but then we will fail
                        // later if we try to look up the fields.
                        debug!("expected struct or union, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprKind::MethodCall(..) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method_id = self.tables.method_map[&method_call].def_id;
                let (def_id, decl_id) = match self.tcx.associated_item(method_id).container {
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

    pub fn get_path_def(&self, id: NodeId) -> Def {
        match self.tcx.map.get(id) {
            Node::NodeTraitRef(tr) => tr.path.def,

            Node::NodeItem(&hir::Item { node: hir::ItemUse(ref path, _), .. }) |
            Node::NodeVisibility(&hir::Visibility::Restricted { ref path, .. }) => path.def,

            Node::NodeExpr(&hir::Expr { node: hir::ExprPath(ref qpath), .. }) |
            Node::NodeExpr(&hir::Expr { node: hir::ExprStruct(ref qpath, ..), .. }) |
            Node::NodePat(&hir::Pat { node: hir::PatKind::Path(ref qpath), .. }) |
            Node::NodePat(&hir::Pat { node: hir::PatKind::Struct(ref qpath, ..), .. }) |
            Node::NodePat(&hir::Pat { node: hir::PatKind::TupleStruct(ref qpath, ..), .. }) => {
                self.tables.qpath_def(qpath, id)
            }

            Node::NodeLocal(&hir::Pat { node: hir::PatKind::Binding(_, def_id, ..), .. }) => {
                Def::Local(def_id)
            }

            Node::NodeTy(&hir::Ty { node: hir::TyPath(ref qpath), .. }) => {
                match *qpath {
                    hir::QPath::Resolved(_, ref path) => path.def,
                    hir::QPath::TypeRelative(..) => {
                        if let Some(ty) = self.analysis.hir_ty_to_ty.get(&id) {
                            if let ty::TyProjection(proj) = ty.sty {
                                for item in self.tcx.associated_items(proj.trait_ref.def_id) {
                                    if item.kind == ty::AssociatedKind::Type {
                                        if item.name == proj.item_name {
                                            return Def::AssociatedTy(item.def_id);
                                        }
                                    }
                                }
                            }
                        }
                        Def::Err
                    }
                }
            }

            _ => Def::Err
        }
    }

    pub fn get_path_data(&self, id: NodeId, path: &ast::Path) -> Option<Data> {
        let def = self.get_path_def(id);
        let sub_span = self.span_utils.span_for_last_ident(path.span);
        filter!(self.span_utils, sub_span, path.span, None);
        match def {
            Def::Upvar(..) |
            Def::Local(..) |
            Def::Static(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::StructCtor(..) |
            Def::VariantCtor(..) => {
                Some(Data::VariableRefData(VariableRefData {
                    name: self.span_utils.snippet(sub_span.unwrap()),
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                    ref_id: def.def_id(),
                }))
            }
            Def::Struct(def_id) |
            Def::Variant(def_id, ..) |
            Def::Union(def_id) |
            Def::Enum(def_id) |
            Def::TyAlias(def_id) |
            Def::AssociatedTy(def_id) |
            Def::Trait(def_id) |
            Def::TyParam(def_id) => {
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
                    let ti = self.tcx.associated_item(decl_id);
                    self.tcx.associated_items(ti.container.id())
                        .find(|item| item.name == ti.name && item.defaultness.has_value())
                        .map(|item| item.def_id)
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
            Def::PrimTy(..) |
            Def::SelfTy(..) |
            Def::Label(..) |
            Def::Macro(..) |
            Def::Err => None,
        }
    }

    pub fn get_field_ref_data(&self,
                              field_ref: &ast::Field,
                              variant: &ty::VariantDef,
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
        match self.get_path_def(ref_id) {
            Def::PrimTy(_) | Def::SelfTy(..) | Def::Err => None,
            def => Some(def.def_id()),
        }
    }

    fn sig_base(&self, item: &ast::Item) -> Signature {
        let text = self.span_utils.signature_string_for_span(item.span);
        let name = item.ident.to_string();
        let ident_start = text.find(&name).expect("Name not in signature?");
        let ident_end = ident_start + name.len();
        Signature {
            span: mk_sp(item.span.lo, item.span.lo + BytePos(text.len() as u32)),
            text: text,
            ident_start: ident_start,
            ident_end: ident_end,
            defs: vec![],
            refs: vec![],
        }
    }

    #[inline]
    pub fn enclosing_scope(&self, id: NodeId) -> NodeId {
        self.tcx.map.get_enclosing_scope(id).unwrap_or(CRATE_NODE_ID)
    }
}

fn make_signature(decl: &ast::FnDecl, generics: &ast::Generics) -> String {
    let mut sig = "fn ".to_owned();
    if !generics.lifetimes.is_empty() || !generics.ty_params.is_empty() {
        sig.push('<');
        sig.push_str(&generics.lifetimes.iter()
                              .map(|l| l.lifetime.name.to_string())
                              .collect::<Vec<_>>()
                              .join(", "));
        if !generics.lifetimes.is_empty() {
            sig.push_str(", ");
        }
        sig.push_str(&generics.ty_params.iter()
                              .map(|l| l.ident.to_string())
                              .collect::<Vec<_>>()
                              .join(", "));
        sig.push_str("> ");
    }
    sig.push('(');
    sig.push_str(&decl.inputs.iter().map(arg_to_string).collect::<Vec<_>>().join(", "));
    sig.push(')');
    match decl.output {
        ast::FunctionRetTy::Default(_) => sig.push_str(" -> ()"),
        ast::FunctionRetTy::Ty(ref t) => sig.push_str(&format!(" -> {}", ty_to_string(t))),
    }

    sig
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

impl<'a> Visitor<'a> for PathCollector {
    fn visit_pat(&mut self, p: &ast::Pat) {
        match p.node {
            PatKind::Struct(ref path, ..) => {
                self.collected_paths.push((p.id, path.clone(),
                                           ast::Mutability::Mutable, recorder::TypeRef));
            }
            PatKind::TupleStruct(ref path, ..) |
            PatKind::Path(_, ref path) => {
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

fn docs_for_attrs(attrs: &[Attribute]) -> String {
    let doc = Symbol::intern("doc");
    let mut result = String::new();

    for attr in attrs {
        if attr.name() == doc {
            if let Some(val) = attr.value_str() {
                if attr.is_sugared_doc {
                    result.push_str(&strip_doc_comment_decoration(&val.as_str()));
                } else {
                    result.push_str(&val.as_str());
                }
                result.push('\n');
            }
        }
    }

    result
}

#[derive(Clone, Copy, Debug, RustcEncodable)]
pub enum Format {
    Csv,
    Json,
    JsonApi,
}

impl Format {
    fn extension(&self) -> &'static str {
        match *self {
            Format::Csv => ".csv",
            Format::Json | Format::JsonApi => ".json",
        }
    }
}

pub fn process_crate<'l, 'tcx>(tcx: TyCtxt<'l, 'tcx, 'tcx>,
                               krate: &ast::Crate,
                               analysis: &'l ty::CrateAnalysis<'tcx>,
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

    if let Err(e) = rustc::util::fs::create_dir_racy(&root_path) {
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

    let save_ctxt = SaveContext {
        tcx: tcx,
        tables: &ty::Tables::empty(),
        analysis: analysis,
        span_utils: SpanUtils::new(&tcx.sess),
    };

    macro_rules! dump {
        ($new_dumper: expr) => {{
            let mut dumper = $new_dumper;
            let mut visitor = DumpVisitor::new(save_ctxt, &mut dumper);

            visitor.dump_crate_info(cratename, krate);
            visit::walk_crate(&mut visitor, krate);
        }}
    }

    match format {
        Format::Csv => dump!(CsvDumper::new(output)),
        Format::Json => dump!(JsonDumper::new(output)),
        Format::JsonApi => dump!(JsonApiDumper::new(output)),
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
