// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::Session;
use middle::ty;
use middle::def;

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use syntax::{attr};
use syntax::ast::{self, NodeId, DefId};
use syntax::ast_util;
use syntax::codemap::*;
use syntax::parse::token::{self, get_ident, keywords};
use syntax::visit::{self, Visitor};
use syntax::print::pprust::ty_to_string;

use util::ppaux;

use self::span_utils::SpanUtils;


mod span_utils;
mod recorder;

mod dump_csv;

pub struct SaveContext<'l, 'tcx: 'l> {
    sess: &'l Session,
    analysis: &'l ty::CrateAnalysis<'tcx>,
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
/// will refere to that variables declaration (by ref_id)).
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


impl<'l, 'tcx: 'l> SaveContext<'l, 'tcx> {
    pub fn new(sess: &'l Session,
               analysis: &'l ty::CrateAnalysis<'tcx>,
               span_utils: SpanUtils<'l>)
               -> SaveContext<'l, 'tcx> {
        SaveContext {
            sess: sess,
            analysis: analysis,
            span_utils: span_utils,
        }
    }

    // List external crates used by the current crate.
    pub fn get_external_crates(&self) -> Vec<CrateData> {
        let mut result = Vec::new();

        self.sess.cstore.iter_crate_data(|n, cmd| {
            result.push(CrateData { name: cmd.name.clone(), number: n });
        });

        result
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Data {
        match item.node {
            ast::ItemFn(..) => {
                let name = self.analysis.ty_cx.map.path_to_string(item.id);
                let qualname = format!("::{}", name);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Fn);

                Data::FunctionData(FunctionData {
                    id: item.id,
                    name: name,
                    qualname: qualname,
                    declaration: None,
                    span: sub_span.unwrap(),
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                })
            }
            ast::ItemStatic(ref typ, mt, ref expr) => {
                let qualname = format!("::{}", self.analysis.ty_cx.map.path_to_string(item.id));

                // If the variable is immutable, save the initialising expression.
                let (value, keyword) = match mt {
                    ast::MutMutable => (String::from("<mutable>"), keywords::Mut),
                    ast::MutImmutable => (self.span_utils.snippet(expr.span), keywords::Static),
                };

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keyword);

                Data::VariableData(VariableData {
                    id: item.id,
                    name: get_ident(item.ident).to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                    value: value,
                    type_value: ty_to_string(&typ),
                })
            }
            ast::ItemConst(ref typ, ref expr) => {
                let qualname = format!("::{}", self.analysis.ty_cx.map.path_to_string(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Const);

                Data::VariableData(VariableData {
                    id: item.id,
                    name: get_ident(item.ident).to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                    value: self.span_utils.snippet(expr.span),
                    type_value: ty_to_string(&typ),
                })
            }
            ast::ItemMod(ref m) => {
                let qualname = format!("::{}", self.analysis.ty_cx.map.path_to_string(item.id));

                let cm = self.sess.codemap();
                let filename = cm.span_to_filename(m.inner);

                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Mod);

                Data::ModData(ModData {
                    id: item.id,
                    name: get_ident(item.ident).to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                    filename: filename,
                })
            },
            ast::ItemEnum(..) => {
                let enum_name = format!("::{}", self.analysis.ty_cx.map.path_to_string(item.id));
                let val = self.span_utils.snippet(item.span);
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Enum);

                Data::EnumData(EnumData {
                    id: item.id,
                    value: val,
                    span: sub_span.unwrap(),
                    qualname: enum_name,
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                })
            },
            ast::ItemImpl(_, _, _, ref trait_ref, ref typ, _) => {
                let mut type_data = None;
                let sub_span;

                let parent = self.analysis.ty_cx.map.get_parent(item.id);

                match typ.node {
                    // Common case impl for a struct or something basic.
                    ast::TyPath(None, ref path) => {
                        sub_span = self.span_utils.sub_span_for_type_name(path.span).unwrap();
                        type_data = self.lookup_ref_id(typ.id).map(|id| TypeRefData {
                            span: sub_span,
                            scope: parent,
                            ref_id: id,
                        });
                    },
                    _ => {
                        // Less useful case, impl for a compound type.
                        let span = typ.span;
                        sub_span = self.span_utils.sub_span_for_type_name(span).unwrap_or(span);
                    }
                }

                let trait_data =
                    trait_ref.as_ref().and_then(|tr| self.get_trait_ref_data(tr, parent));

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

    // FIXME: we ought to be able to get the parent id ourselves, but we can't
    // for now.
    pub fn get_field_data(&self, field: &ast::StructField, parent: NodeId) -> Option<Data> {
        match field.node.kind {
            ast::NamedField(ident, _) => {
                let name = get_ident(ident);
                let qualname = format!("::{}::{}",
                                       self.analysis.ty_cx.map.path_to_string(parent),
                                       name);
                let typ = ppaux::ty_to_string(&self.analysis.ty_cx,
                                              *self.analysis.ty_cx.node_types()
                                                  .get(&field.node.id).unwrap());
                let sub_span = self.span_utils.sub_span_before_token(field.span, token::Colon);
                Some(Data::VariableData(VariableData {
                    id: field.node.id,
                    name: get_ident(ident).to_string(),
                    qualname: qualname,
                    span: sub_span.unwrap(),
                    scope: parent,
                    value: "".to_owned(),
                    type_value: typ,
                }))
            },
            _ => None,
        }
    }

    // FIXME: we ought to be able to get the parent id ourselves, but we can't
    // for now.
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
                let ty = &ty::expr_ty_adjusted(&self.analysis.ty_cx, &sub_ex).sty;
                match *ty {
                    ty::TyStruct(def_id, _) => {
                        let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, def_id);
                        for f in &fields {
                            if f.name == ident.node.name {
                                let sub_span = self.span_utils.span_for_last_ident(expr.span);
                                return Some(Data::VariableRefData(VariableRefData {
                                    name: get_ident(ident.node).to_string(),
                                    span: sub_span.unwrap(),
                                    scope: self.analysis.ty_cx.map.get_parent(expr.id),
                                    ref_id: f.id,
                                }));
                            }
                        }

                        self.sess.span_bug(expr.span,
                                           &format!("Couldn't find field {} on {:?}",
                                                    &get_ident(ident.node),
                                                    ty))
                    }
                    _ => {
                        debug!("Expected struct type, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprStruct(ref path, _, _) => {
                let ty = &ty::expr_ty_adjusted(&self.analysis.ty_cx, expr).sty;
                match *ty {
                    ty::TyStruct(def_id, _) => {
                        let sub_span = self.span_utils.span_for_last_ident(path.span);
                        Some(Data::TypeRefData(TypeRefData {
                            span: sub_span.unwrap(),
                            scope: self.analysis.ty_cx.map.get_parent(expr.id),
                            ref_id: def_id,
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
            _ => {
                // FIXME
                unimplemented!();
            }
        }
    }

    pub fn get_field_ref_data(&self,
                              field_ref: &ast::Field,
                              struct_id: DefId,
                              parent: NodeId)
                              -> VariableRefData {
        let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, struct_id);
        let field_name = get_ident(field_ref.ident.node).to_string();
        for f in &fields {
            if f.name == field_ref.ident.node.name {
                // We don't really need a sub-span here, but no harm done
                let sub_span = self.span_utils.span_for_last_ident(field_ref.ident.span);
                return VariableRefData {
                    name: field_name,
                    span: sub_span.unwrap(),
                    scope: parent,
                    ref_id: f.id,
                };
            }
        }

        self.sess.span_bug(field_ref.span,
                           &format!("Couldn't find field {}", field_name));
    }

    pub fn get_data_for_id(&self, _id: &NodeId) -> Data {
        // FIXME
        unimplemented!();
    }

    fn lookup_ref_id(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.analysis.ty_cx.def_map.borrow().contains_key(&ref_id) {
            self.sess.bug(&format!("def_map has no key for {} in lookup_type_ref",
                                  ref_id));
        }
        let def = self.analysis.ty_cx.def_map.borrow().get(&ref_id).unwrap().full_def();
        match def {
            def::DefPrimTy(_) => None,
            _ => Some(def.def_id()),
        }
    }

}

// An AST visitor for collecting paths from patterns.
struct PathCollector {
    // The Row field identifies the kind of pattern.
    collected_paths: Vec<(NodeId, ast::Path, ast::Mutability, recorder::Row)>,
}

impl PathCollector {
    fn new() -> PathCollector {
        PathCollector {
            collected_paths: vec![],
        }
    }
}

impl<'v> Visitor<'v> for PathCollector {
    fn visit_pat(&mut self, p: &ast::Pat) {
        if generated_code(p.span) {
            return;
        }

        match p.node {
            ast::PatStruct(ref path, _, _) => {
                self.collected_paths.push((p.id,
                                           path.clone(),
                                           ast::MutMutable,
                                           recorder::TypeRef));
            }
            ast::PatEnum(ref path, _) |
            ast::PatQPath(_, ref path) => {
                self.collected_paths.push((p.id, path.clone(), ast::MutMutable, recorder::VarRef));
            }
            ast::PatIdent(bm, ref path1, _) => {
                debug!("PathCollector, visit ident in pat {}: {:?} {:?}",
                       token::get_ident(path1.node),
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

#[allow(deprecated)]
pub fn process_crate(sess: &Session,
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
        },
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

    match fs::create_dir_all(&root_path) {
        Err(e) => sess.err(&format!("Could not create directory {}: {}",
                           root_path.display(), e)),
        _ => (),
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
            sess.fatal(&format!("Could not open {}: {}", disp, e));
        }
    };
    root_path.pop();

    let mut visitor = dump_csv::DumpCsvVisitor::new(sess, analysis, output_file);

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
fn generated_code(span: Span) -> bool {
    span.expn_id != NO_EXPANSION || span  == DUMMY_SP
}
