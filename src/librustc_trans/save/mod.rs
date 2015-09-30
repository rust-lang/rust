// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;
use middle::def;
use middle::def_id::DefId;

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use rustc_front;
use rustc::front::map::NodeItem;
use rustc_front::{hir, lowering};

use syntax::attr;
use syntax::ast::{self, NodeId};
use syntax::ast_util;
use syntax::codemap::*;
use syntax::parse::token::{self, keywords};
use syntax::visit::{self, Visitor};
use syntax::print::pprust::ty_to_string;

use self::span_utils::SpanUtils;


pub mod span_utils;
pub mod recorder;

mod dump_csv;

pub struct SaveContext<'l, 'tcx: 'l> {
    tcx: &'l ty::ctxt<'tcx>,
    lcx: &'l lowering::LoweringContext<'l>,
    span_utils: SpanUtils<'l>,
}

pub struct CrateData {
    pub name: String,
    pub number: u32,
}

/// Data for any entity in the Rust language. The actual data contained varied
/// with the kind of entity being queried. See the nested structs for details.
#[derive(Debug)]
pub enum Data {
    /// Data for all kinds of functions and methods.
    FunctionData(FunctionData),
    /// Data for local and global variables (consts and statics), and fields.
    VariableData(VariableData),
    /// Data for modules.
    ModData(ModData),
    /// Data for Enums.
    EnumData(EnumData),
    /// Data for impls.
    ImplData(ImplData),

    /// Data for the use of some variable (e.g., the use of a local variable, which
    /// will refere to that variables declaration).
    VariableRefData(VariableRefData),
    /// Data for a reference to a type or trait.
    TypeRefData(TypeRefData),
    /// Data for a reference to a module.
    ModRefData(ModRefData),
    /// Data about a function call.
    FunctionCallData(FunctionCallData),
    /// Data about a method call.
    MethodCallData(MethodCallData),
}

/// Data for all kinds of functions and methods.
#[derive(Debug)]
pub struct FunctionData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub declaration: Option<DefId>,
    pub span: Span,
    pub scope: NodeId,
}

/// Data for local and global variables (consts and statics).
#[derive(Debug)]
pub struct VariableData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub value: String,
    pub type_value: String,
}

/// Data for modules.
#[derive(Debug)]
pub struct ModData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub filename: String,
}

/// Data for enum declarations.
#[derive(Debug)]
pub struct EnumData {
    pub id: NodeId,
    pub value: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
}

#[derive(Debug)]
pub struct ImplData {
    pub id: NodeId,
    pub span: Span,
    pub scope: NodeId,
    // FIXME: I'm not really sure inline data is the best way to do this. Seems
    // OK in this case, but generalising leads to returning chunks of AST, which
    // feels wrong.
    pub trait_ref: Option<TypeRefData>,
    pub self_ref: Option<TypeRefData>,
}

/// Data for the use of some item (e.g., the use of a local variable, which
/// will refer to that variables declaration (by ref_id)).
#[derive(Debug)]
pub struct VariableRefData {
    pub name: String,
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

/// Data for a reference to a type or trait.
#[derive(Debug)]
pub struct TypeRefData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

/// Data for a reference to a module.
#[derive(Debug)]
pub struct ModRefData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

/// Data about a function call.
#[derive(Debug)]
pub struct FunctionCallData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

/// Data about a method call.
#[derive(Debug)]
pub struct MethodCallData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: Option<DefId>,
    pub decl_id: Option<DefId>,
}



impl<'l, 'tcx: 'l> SaveContext<'l, 'tcx> {
    pub fn new(tcx: &'l ty::ctxt<'tcx>,
               lcx: &'l lowering::LoweringContext<'l>)
               -> SaveContext<'l, 'tcx> {
        let span_utils = SpanUtils::new(&tcx.sess);
        SaveContext::from_span_utils(tcx, lcx, span_utils)
    }

    pub fn from_span_utils(tcx: &'l ty::ctxt<'tcx>,
                           lcx: &'l lowering::LoweringContext<'l>,
                           span_utils: SpanUtils<'l>)
                           -> SaveContext<'l, 'tcx> {
        SaveContext {
            tcx: tcx,
            lcx: lcx,
            span_utils: span_utils,
        }
    }

    // List external crates used by the current crate.
    pub fn get_external_crates(&self) -> Vec<CrateData> {
        let mut result = Vec::new();

        self.tcx.sess.cstore.iter_crate_data(|n, cmd| {
            result.push(CrateData {
                name: cmd.name.clone(),
                number: n,
            });
        });

        result
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Data {
        match item.node {
            ast::ItemFn(..) => {
                let name = self.tcx.map.path_to_string(item.id);
                let qualname = format!("::{}", name);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Fn);

                Data::FunctionData(FunctionData {
                    id: item.id,
                    name: name,
                    qualname: qualname,
                    declaration: None,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                })
            }
            ast::ItemStatic(ref typ, mt, ref expr) => {
                let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));

                // If the variable is immutable, save the initialising expression.
                let (value, keyword) = match mt {
                    ast::MutMutable => (String::from("<mutable>"), keywords::Mut),
                    ast::MutImmutable => (self.span_utils.snippet(expr.span), keywords::Static),
                };

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keyword);

                Data::VariableData(VariableData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    value: value,
                    type_value: ty_to_string(&typ),
                })
            }
            ast::ItemConst(ref typ, ref expr) => {
                let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Const);

                Data::VariableData(VariableData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    value: self.span_utils.snippet(expr.span),
                    type_value: ty_to_string(&typ),
                })
            }
            ast::ItemMod(ref m) => {
                let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));

                let cm = self.tcx.sess.codemap();
                let filename = cm.span_to_filename(m.inner);

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Mod);

                Data::ModData(ModData {
                    id: item.id,
                    name: item.ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(item.id),
                    filename: filename,
                })
            }
            ast::ItemEnum(..) => {
                let enum_name = format!("::{}", self.tcx.map.path_to_string(item.id));
                let val = self.span_utils.snippet(item.span);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Enum);

                Data::EnumData(EnumData {
                    id: item.id,
                    value: val,
                    span: sub_span.unwrap(),
                    qualname: enum_name,
                    scope: self.enclosing_scope(item.id),
                })
            }
            ast::ItemImpl(_, _, _, ref trait_ref, ref typ, _) => {
                let mut type_data = None;
                let sub_span;

                let parent = self.enclosing_scope(item.id);

                match typ.node {
                    // Common case impl for a struct or something basic.
                    ast::TyPath(None, ref path) => {
                        sub_span = self.span_utils.sub_span_for_type_name(path.span).unwrap();
                        type_data = self.lookup_ref_id(typ.id).map(|id| {
                            TypeRefData {
                                span: sub_span,
                                scope: parent,
                                ref_id: id,
                            }
                        });
                    }
                    _ => {
                        // Less useful case, impl for a compound type.
                        let span = typ.span;
                        sub_span = self.span_utils.sub_span_for_type_name(span).unwrap_or(span);
                    }
                }

                let trait_data = trait_ref.as_ref()
                                          .and_then(|tr| self.get_trait_ref_data(tr, parent));

                Data::ImplData(ImplData {
                    id: item.id,
                    span: sub_span,
                    scope: parent,
                    trait_ref: trait_data,
                    self_ref: type_data,
                })
            }
            _ => {
                // FIXME
                unimplemented!();
            }
        }
    }

    pub fn get_field_data(&self, field: &ast::StructField, scope: NodeId) -> Option<VariableData> {
        match field.node.kind {
            ast::NamedField(ident, _) => {
                let qualname = format!("::{}::{}", self.tcx.map.path_to_string(scope), ident);
                let typ = self.tcx.node_types().get(&field.node.id).unwrap().to_string();
                let sub_span = self.span_utils.sub_span_before_token(field.span, token::Colon);
                Some(VariableData {
                    id: field.node.id,
                    name: ident.to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: scope,
                    value: "".to_owned(),
                    type_value: typ,
                })
            }
            _ => None,
        }
    }

    // FIXME would be nice to take a MethodItem here, but the ast provides both
    // trait and impl flavours, so the caller must do the disassembly.
    pub fn get_method_data(&self, id: ast::NodeId, name: ast::Name, span: Span) -> FunctionData {
        // The qualname for a method is the trait name or name of the struct in an impl in
        // which the method is declared in, followed by the method's name.
        let qualname = match self.tcx.impl_of_method(self.tcx.map.local_def_id(id)) {
            Some(impl_id) => match self.tcx.map.get_if_local(impl_id) {
                Some(NodeItem(item)) => {
                    match item.node {
                        hir::ItemImpl(_, _, _, _, ref ty, _) => {
                            let mut result = String::from("<");
                            result.push_str(&rustc_front::print::pprust::ty_to_string(&**ty));

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
                            self.tcx.sess.span_bug(span,
                                                   &format!("Container {:?} for method {} not \
                                                             an impl?",
                                                            impl_id,
                                                            id));
                        }
                    }
                }
                r => {
                    self.tcx.sess.span_bug(span,
                                           &format!("Container {:?} for method {} is not a node \
                                                     item {:?}",
                                                    impl_id,
                                                    id,
                                                    r));
                }
            },
            None => match self.tcx.trait_of_item(self.tcx.map.local_def_id(id)) {
                Some(def_id) => {
                    match self.tcx.map.get_if_local(def_id) {
                        Some(NodeItem(_)) => {
                            format!("::{}", self.tcx.item_path_str(def_id))
                        }
                        r => {
                            self.tcx.sess.span_bug(span,
                                                   &format!("Could not find container {:?} for \
                                                             method {}, got {:?}",
                                                            def_id,
                                                            id,
                                                            r));
                        }
                    }
                }
                None => {
                    self.tcx.sess.span_bug(span,
                                           &format!("Could not find container for method {}", id));
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

        FunctionData {
            id: id,
            name: name.to_string(),
            qualname: qualname,
            declaration: decl_id,
            span: sub_span.unwrap(),
            scope: self.enclosing_scope(id),
        }
    }

    pub fn get_trait_ref_data(&self,
                              trait_ref: &ast::TraitRef,
                              parent: NodeId)
                              -> Option<TypeRefData> {
        self.lookup_ref_id(trait_ref.ref_id).map(|def_id| {
            let span = trait_ref.path.span;
            let sub_span = self.span_utils.sub_span_for_type_name(span).unwrap_or(span);
            TypeRefData {
                span: sub_span,
                scope: parent,
                ref_id: def_id,
            }
        })
    }

    pub fn get_expr_data(&self, expr: &ast::Expr) -> Option<Data> {
        match expr.node {
            ast::ExprField(ref sub_ex, ident) => {
                let hir_node = lowering::lower_expr(self.lcx, sub_ex);
                let ty = &self.tcx.expr_ty_adjusted(&hir_node).sty;
                match *ty {
                    ty::TyStruct(def, _) => {
                        let f = def.struct_variant().field_named(ident.node.name);
                        let sub_span = self.span_utils.span_for_last_ident(expr.span);
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
            ast::ExprStruct(ref path, _, _) => {
                let hir_node = lowering::lower_expr(self.lcx, expr);
                let ty = &self.tcx.expr_ty_adjusted(&hir_node).sty;
                match *ty {
                    ty::TyStruct(def, _) => {
                        let sub_span = self.span_utils.span_for_last_ident(path.span);
                        Some(Data::TypeRefData(TypeRefData {
                            span: sub_span.unwrap(),
                            scope: self.enclosing_scope(expr.id),
                            ref_id: def.did,
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
            ast::ExprMethodCall(..) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method_id = self.tcx.tables.borrow().method_map[&method_call].def_id;
                let (def_id, decl_id) = match self.tcx.impl_or_trait_item(method_id).container() {
                    ty::ImplContainer(_) => (Some(method_id), None),
                    ty::TraitContainer(_) => (None, Some(method_id)),
                };
                let sub_span = self.span_utils.sub_span_for_meth_name(expr.span);
                let parent = self.enclosing_scope(expr.id);
                Some(Data::MethodCallData(MethodCallData {
                    span: sub_span.unwrap(),
                    scope: parent,
                    ref_id: def_id,
                    decl_id: decl_id,
                }))
            }
            ast::ExprPath(_, ref path) => {
                self.get_path_data(expr.id, path)
            }
            _ => {
                // FIXME
                unimplemented!();
            }
        }
    }

    pub fn get_path_data(&self, id: NodeId, path: &ast::Path) -> Option<Data> {
        let def_map = self.tcx.def_map.borrow();
        if !def_map.contains_key(&id) {
            self.tcx.sess.span_bug(path.span,
                                   &format!("def_map has no key for {} in visit_expr", id));
        }
        let def = def_map.get(&id).unwrap().full_def();
        let sub_span = self.span_utils.span_for_last_ident(path.span);
        match def {
            def::DefUpvar(..) |
            def::DefLocal(..) |
            def::DefStatic(..) |
            def::DefConst(..) |
            def::DefAssociatedConst(..) |
            def::DefVariant(..) => {
                Some(Data::VariableRefData(VariableRefData {
                    name: self.span_utils.snippet(sub_span.unwrap()),
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                    ref_id: def.def_id(),
                }))
            }
            def::DefStruct(def_id) |
            def::DefTy(def_id, _) |
            def::DefTrait(def_id) |
            def::DefTyParam(_, _, def_id, _) => {
                Some(Data::TypeRefData(TypeRefData {
                    span: sub_span.unwrap(),
                    ref_id: def_id,
                    scope: self.enclosing_scope(id),
                }))
            }
            def::DefMethod(decl_id) => {
                let sub_span = self.span_utils.sub_span_for_meth_name(path.span);
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
            def::DefFn(def_id, _) => {
                Some(Data::FunctionCallData(FunctionCallData {
                    ref_id: def_id,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
                }))
            }
            def::DefMod(def_id) => {
                Some(Data::ModRefData(ModRefData {
                    ref_id: def_id,
                    span: sub_span.unwrap(),
                    scope: self.enclosing_scope(id),
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
                              -> VariableRefData {
        let f = variant.field_named(field_ref.ident.node.name);
        // We don't really need a sub-span here, but no harm done
        let sub_span = self.span_utils.span_for_last_ident(field_ref.ident.span);
        VariableRefData {
            name: field_ref.ident.node.to_string(),
            span: sub_span.unwrap(),
            scope: parent,
            ref_id: f.did,
        }
    }

    pub fn get_data_for_id(&self, _id: &NodeId) -> Data {
        // FIXME
        unimplemented!();
    }

    fn lookup_ref_id(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.tcx.def_map.borrow().contains_key(&ref_id) {
            self.tcx.sess.bug(&format!("def_map has no key for {} in lookup_type_ref",
                                       ref_id));
        }
        let def = self.tcx.def_map.borrow().get(&ref_id).unwrap().full_def();
        match def {
            def::DefPrimTy(_) | def::DefSelfTy(..) => None,
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
        if generated_code(p.span) {
            return;
        }

        match p.node {
            ast::PatStruct(ref path, _, _) => {
                self.collected_paths.push((p.id, path.clone(), ast::MutMutable, recorder::TypeRef));
            }
            ast::PatEnum(ref path, _) |
            ast::PatQPath(_, ref path) => {
                self.collected_paths.push((p.id, path.clone(), ast::MutMutable, recorder::VarRef));
            }
            ast::PatIdent(bm, ref path1, _) => {
                debug!("PathCollector, visit ident in pat {}: {:?} {:?}",
                       path1.node,
                       p.span,
                       path1.span);
                let immut = match bm {
                    // Even if the ref is mut, you can't change the ref, only
                    // the data pointed at, so showing the initialising expression
                    // is still worthwhile.
                    ast::BindByRef(_) => ast::MutImmutable,
                    ast::BindByValue(mt) => mt,
                };
                // collect path for either visit_local or visit_arm
                let path = ast_util::ident_to_path(path1.span, path1.node);
                self.collected_paths.push((p.id, path, immut, recorder::VarRef));
            }
            _ => {}
        }
        visit::walk_pat(self, p);
    }
}

pub fn process_crate<'l, 'tcx>(tcx: &'l ty::ctxt<'tcx>,
                               lcx: &'l lowering::LoweringContext<'l>,
                               krate: &ast::Crate,
                               analysis: &ty::CrateAnalysis,
                               odir: Option<&Path>) {
    if generated_code(krate.span) {
        return;
    }

    assert!(analysis.glob_map.is_some());
    let cratename = match attr::find_crate_name(&krate.attrs) {
        Some(name) => name.to_string(),
        None => {
            info!("Could not find crate name, using 'unknown_crate'");
            String::from("unknown_crate")
        }
    };

    info!("Dumping crate {}", cratename);

    // find a path to dump our data to
    let mut root_path = match env::var_os("DXR_RUST_TEMP_FOLDER") {
        Some(val) => PathBuf::from(val),
        None => match odir {
            Some(val) => val.join("dxr"),
            None => PathBuf::from("dxr-temp"),
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
    let mut out_name = cratename.clone();
    out_name.push_str(".csv");
    root_path.push(&out_name);
    let output_file = match File::create(&root_path) {
        Ok(f) => box f,
        Err(e) => {
            let disp = root_path.display();
            tcx.sess.fatal(&format!("Could not open {}: {}", disp, e));
        }
    };
    root_path.pop();

    let mut visitor = dump_csv::DumpCsvVisitor::new(tcx, lcx, analysis, output_file);

    visitor.dump_crate_info(&cratename, krate);
    visit::walk_crate(&mut visitor, krate);
}

// Utility functions for the module.

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

// If the expression is a macro expansion or other generated code, run screaming
// and don't index.
pub fn generated_code(span: Span) -> bool {
    span.expn_id != NO_EXPANSION || span == DUMMY_SP
}
