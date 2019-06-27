#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(nll)]
#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]
#![allow(unused_attributes)]

#![recursion_limit="256"]


mod json_dumper;
mod dump_visitor;
#[macro_use]
mod span_utils;
mod sig;

use rustc::hir;
use rustc::hir::def::{CtorOf, Res, DefKind as HirDefKind};
use rustc::hir::Node;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::privacy::AccessLevels;
use rustc::middle::cstore::ExternCrate;
use rustc::session::config::{CrateType, Input, OutputType};
use rustc::ty::{self, DefIdTree, TyCtxt};
use rustc::{bug, span_bug};
use rustc_codegen_utils::link::{filename_for_metadata, out_filename};

use std::cell::Cell;
use std::default::Default;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use syntax::ast::{self, Attribute, DUMMY_NODE_ID, NodeId, PatKind};
use syntax::source_map::Spanned;
use syntax::parse::lexer::comments::strip_doc_comment_decoration;
use syntax::print::pprust;
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{arg_to_string, ty_to_string};
use syntax::source_map::MacroAttribute;
use syntax_pos::*;

use json_dumper::JsonDumper;
use dump_visitor::DumpVisitor;
use span_utils::SpanUtils;

use rls_data::{Def, DefKind, ExternalCrateData, GlobalCrateId, MacroRef, Ref, RefKind, Relation,
               RelationKind, SpanData, Impl, ImplKind};
use rls_data::config::Config;

use log::{debug, error, info};


pub struct SaveContext<'l, 'tcx> {
    tcx: TyCtxt<'tcx>,
    tables: &'l ty::TypeckTables<'tcx>,
    access_levels: &'l AccessLevels,
    span_utils: SpanUtils<'tcx>,
    config: Config,
    impl_counter: Cell<u32>,
}

#[derive(Debug)]
pub enum Data {
    RefData(Ref),
    DefData(Def),
    RelationData(Relation, Impl),
}

impl<'l, 'tcx> SaveContext<'l, 'tcx> {
    fn span_from_span(&self, span: Span) -> SpanData {
        use rls_span::{Column, Row};

        let cm = self.tcx.sess.source_map();
        let start = cm.lookup_char_pos(span.lo());
        let end = cm.lookup_char_pos(span.hi());

        SpanData {
            file_name: start.file.name.to_string().into(),
            byte_start: span.lo().0,
            byte_end: span.hi().0,
            line_start: Row::new_one_indexed(start.line as u32),
            line_end: Row::new_one_indexed(end.line as u32),
            column_start: Column::new_one_indexed(start.col.0 as u32 + 1),
            column_end: Column::new_one_indexed(end.col.0 as u32 + 1),
        }
    }

    // Returns path to the compilation output (e.g., libfoo-12345678.rmeta)
    pub fn compilation_output(&self, crate_name: &str) -> PathBuf {
        let sess = &self.tcx.sess;
        // Save-analysis is emitted per whole session, not per each crate type
        let crate_type = sess.crate_types.borrow()[0];
        let outputs = &*self.tcx.output_filenames(LOCAL_CRATE);

        if outputs.outputs.contains_key(&OutputType::Metadata) {
            filename_for_metadata(sess, crate_name, outputs)
        } else if outputs.outputs.should_codegen() {
            out_filename(sess, crate_type, outputs, crate_name)
        } else {
            // Otherwise it's only a DepInfo, in which case we return early and
            // not even reach the analysis stage.
            unreachable!()
        }
    }

    // List external crates used by the current crate.
    pub fn get_external_crates(&self) -> Vec<ExternalCrateData> {
        let mut result = Vec::with_capacity(self.tcx.crates().len());

        for &n in self.tcx.crates().iter() {
            let span = match self.tcx.extern_crate(n.as_def_id()) {
                Some(&ExternCrate { span, .. }) => span,
                None => {
                    debug!("Skipping crate {}, no data", n);
                    continue;
                }
            };
            let lo_loc = self.span_utils.sess.source_map().lookup_char_pos(span.lo());
            result.push(ExternalCrateData {
                // FIXME: change file_name field to PathBuf in rls-data
                // https://github.com/nrc/rls-data/issues/7
                file_name: self.span_utils.make_filename_string(&lo_loc.file),
                num: n.as_u32(),
                id: GlobalCrateId {
                    name: self.tcx.crate_name(n).to_string(),
                    disambiguator: self.tcx.crate_disambiguator(n).to_fingerprint().as_value(),
                },
            });
        }

        result
    }

    pub fn get_extern_item_data(&self, item: &ast::ForeignItem) -> Option<Data> {
        let qualname = format!("::{}",
            self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
        match item.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                filter!(self.span_utils, item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::ForeignFunction,
                    id: id_from_node_id(item.id, self),
                    span: self.span_from_span(item.ident.span),
                    name: item.ident.to_string(),
                    qualname,
                    value: make_signature(decl, generics),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::foreign_item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ForeignItemKind::Static(ref ty, _) => {
                filter!(self.span_utils, item.ident.span);

                let id = id_from_node_id(item.id, self);
                let span = self.span_from_span(item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::ForeignStatic,
                    id,
                    span,
                    name: item.ident.to_string(),
                    qualname,
                    value: ty_to_string(ty),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::foreign_item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            // FIXME(plietar): needs a new DefKind in rls-data
            ast::ForeignItemKind::Ty => None,
            ast::ForeignItemKind::Macro(..) => None,
        }
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Option<Data> {
        match item.node {
            ast::ItemKind::Fn(ref decl, .., ref generics, _) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
                filter!(self.span_utils, item.ident.span);
                Some(Data::DefData(Def {
                    kind: DefKind::Function,
                    id: id_from_node_id(item.id, self),
                    span: self.span_from_span(item.ident.span),
                    name: item.ident.to_string(),
                    qualname,
                    value: make_signature(decl, generics),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ItemKind::Static(ref typ, ..) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));

                filter!(self.span_utils, item.ident.span);

                let id = id_from_node_id(item.id, self);
                let span = self.span_from_span(item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::Static,
                    id,
                    span,
                    name: item.ident.to_string(),
                    qualname,
                    value: ty_to_string(&typ),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ItemKind::Const(ref typ, _) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
                filter!(self.span_utils, item.ident.span);

                let id = id_from_node_id(item.id, self);
                let span = self.span_from_span(item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::Const,
                    id,
                    span,
                    name: item.ident.to_string(),
                    qualname,
                    value: ty_to_string(typ),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ItemKind::Mod(ref m) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));

                let cm = self.tcx.sess.source_map();
                let filename = cm.span_to_filename(m.inner);

                filter!(self.span_utils, item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::Mod,
                    id: id_from_node_id(item.id, self),
                    name: item.ident.to_string(),
                    qualname,
                    span: self.span_from_span(item.ident.span),
                    value: filename.to_string(),
                    parent: None,
                    children: m.items
                        .iter()
                        .map(|i| id_from_node_id(i.id, self))
                        .collect(),
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ItemKind::Enum(ref def, _) => {
                let name = item.ident.to_string();
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
                filter!(self.span_utils, item.ident.span);
                let variants_str = def.variants
                    .iter()
                    .map(|v| v.node.ident.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let value = format!("{}::{{{}}}", name, variants_str);
                Some(Data::DefData(Def {
                    kind: DefKind::Enum,
                    id: id_from_node_id(item.id, self),
                    span: self.span_from_span(item.ident.span),
                    name,
                    qualname,
                    value,
                    parent: None,
                    children: def.variants
                        .iter()
                        .map(|v| id_from_node_id(v.node.id, self))
                        .collect(),
                    decl_id: None,
                    docs: self.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(item.attrs.clone(), self),
                }))
            }
            ast::ItemKind::Impl(.., ref trait_ref, ref typ, ref impls) => {
                if let ast::TyKind::Path(None, ref path) = typ.node {
                    // Common case impl for a struct or something basic.
                    if generated_code(path.span) {
                        return None;
                    }
                    let sub_span = path.segments.last().unwrap().ident.span;
                    filter!(self.span_utils, sub_span);

                    let impl_id = self.next_impl_id();
                    let span = self.span_from_span(sub_span);

                    let type_data = self.lookup_ref_id(typ.id);
                    type_data.map(|type_data| {
                        Data::RelationData(Relation {
                            kind: RelationKind::Impl {
                                id: impl_id,
                            },
                            span: span.clone(),
                            from: id_from_def_id(type_data),
                            to: trait_ref
                                .as_ref()
                                .and_then(|t| self.lookup_ref_id(t.ref_id))
                                .map(id_from_def_id)
                                .unwrap_or_else(|| null_id()),
                        },
                        Impl {
                            id: impl_id,
                            kind: match *trait_ref {
                                Some(_) => ImplKind::Direct,
                                None => ImplKind::Inherent,
                            },
                            span: span,
                            value: String::new(),
                            parent: None,
                            children: impls
                                .iter()
                                .map(|i| id_from_node_id(i.id, self))
                                .collect(),
                            docs: String::new(),
                            sig: None,
                            attributes: vec![],
                        })
                    })
                } else {
                    None
                }
            }
            _ => {
                // FIXME
                bug!();
            }
        }
    }

    pub fn get_field_data(&self, field: &ast::StructField, scope: NodeId) -> Option<Def> {
        if let Some(ident) = field.ident {
            let name = ident.to_string();
            let qualname = format!("::{}::{}",
                self.tcx.def_path_str(self.tcx.hir().local_def_id(scope)),
                ident);
            filter!(self.span_utils, ident.span);
            let def_id = self.tcx.hir().local_def_id(field.id);
            let typ = self.tcx.type_of(def_id).to_string();


            let id = id_from_node_id(field.id, self);
            let span = self.span_from_span(ident.span);

            Some(Def {
                kind: DefKind::Field,
                id,
                span,
                name,
                qualname,
                value: typ,
                parent: Some(id_from_node_id(scope, self)),
                children: vec![],
                decl_id: None,
                docs: self.docs_for_attrs(&field.attrs),
                sig: sig::field_signature(field, self),
                attributes: lower_attributes(field.attrs.clone(), self),
            })
        } else {
            None
        }
    }

    // FIXME would be nice to take a MethodItem here, but the ast provides both
    // trait and impl flavours, so the caller must do the disassembly.
    pub fn get_method_data(&self, id: ast::NodeId, ident: ast::Ident, span: Span) -> Option<Def> {
        // The qualname for a method is the trait name or name of the struct in an impl in
        // which the method is declared in, followed by the method's name.
        let (qualname, parent_scope, decl_id, docs, attributes) =
            match self.tcx.impl_of_method(self.tcx.hir().local_def_id(id)) {
                Some(impl_id) => match self.tcx.hir().get_if_local(impl_id) {
                    Some(Node::Item(item)) => match item.node {
                        hir::ItemKind::Impl(.., ref ty, _) => {
                            let mut qualname = String::from("<");
                            qualname.push_str(&self.tcx.hir().hir_to_pretty_string(ty.hir_id));

                            let trait_id = self.tcx.trait_id_of_impl(impl_id);
                            let mut decl_id = None;
                            let mut docs = String::new();
                            let mut attrs = vec![];
                            let hir_id = self.tcx.hir().node_to_hir_id(id);
                            if let Some(Node::ImplItem(item)) =
                                self.tcx.hir().find(hir_id)
                            {
                                docs = self.docs_for_attrs(&item.attrs);
                                attrs = item.attrs.to_vec();
                            }

                            if let Some(def_id) = trait_id {
                                // A method in a trait impl.
                                qualname.push_str(" as ");
                                qualname.push_str(&self.tcx.def_path_str(def_id));
                                self.tcx
                                    .associated_items(def_id)
                                    .find(|item| item.ident.name == ident.name)
                                    .map(|item| decl_id = Some(item.def_id));
                            }
                            qualname.push_str(">");

                            (qualname, trait_id, decl_id, docs, attrs)
                        }
                        _ => {
                            span_bug!(
                                span,
                                "Container {:?} for method {} not an impl?",
                                impl_id,
                                id
                            );
                        }
                    },
                    r => {
                        span_bug!(
                            span,
                            "Container {:?} for method {} is not a node item {:?}",
                            impl_id,
                            id,
                            r
                        );
                    }
                },
                None => match self.tcx.trait_of_item(self.tcx.hir().local_def_id(id)) {
                    Some(def_id) => {
                        let mut docs = String::new();
                        let mut attrs = vec![];
                        let hir_id = self.tcx.hir().node_to_hir_id(id);

                        if let Some(Node::TraitItem(item)) = self.tcx.hir().find(hir_id) {
                            docs = self.docs_for_attrs(&item.attrs);
                            attrs = item.attrs.to_vec();
                        }

                        (
                            format!("::{}", self.tcx.def_path_str(def_id)),
                            Some(def_id),
                            None,
                            docs,
                            attrs,
                        )
                    }
                    None => {
                        debug!("Could not find container for method {} at {:?}", id, span);
                        // This is not necessarily a bug, if there was a compilation error,
                        // the tables we need might not exist.
                        return None;
                    }
                },
            };

        let qualname = format!("{}::{}", qualname, ident.name);

        filter!(self.span_utils, ident.span);

        Some(Def {
            kind: DefKind::Method,
            id: id_from_node_id(id, self),
            span: self.span_from_span(ident.span),
            name: ident.name.to_string(),
            qualname,
            // FIXME you get better data here by using the visitor.
            value: String::new(),
            parent: parent_scope.map(|id| id_from_def_id(id)),
            children: vec![],
            decl_id: decl_id.map(|id| id_from_def_id(id)),
            docs,
            sig: None,
            attributes: lower_attributes(attributes, self),
        })
    }

    pub fn get_trait_ref_data(&self, trait_ref: &ast::TraitRef) -> Option<Ref> {
        self.lookup_ref_id(trait_ref.ref_id).and_then(|def_id| {
            let span = trait_ref.path.span;
            if generated_code(span) {
                return None;
            }
            let sub_span = trait_ref.path.segments.last().unwrap().ident.span;
            filter!(self.span_utils, sub_span);
            let span = self.span_from_span(sub_span);
            Some(Ref {
                kind: RefKind::Type,
                span,
                ref_id: id_from_def_id(def_id),
            })
        })
    }

    pub fn get_expr_data(&self, expr: &ast::Expr) -> Option<Data> {
        let expr_hir_id = self.tcx.hir().node_to_hir_id(expr.id);
        let hir_node = self.tcx.hir().expect_expr(expr_hir_id);
        let ty = self.tables.expr_ty_adjusted_opt(&hir_node);
        if ty.is_none() || ty.unwrap().sty == ty::Error {
            return None;
        }
        match expr.node {
            ast::ExprKind::Field(ref sub_ex, ident) => {
                let sub_ex_hir_id = self.tcx.hir().node_to_hir_id(sub_ex.id);
                let hir_node = match self.tcx.hir().find(sub_ex_hir_id) {
                    Some(Node::Expr(expr)) => expr,
                    _ => {
                        debug!(
                            "Missing or weird node for sub-expression {} in {:?}",
                            sub_ex.id,
                            expr
                        );
                        return None;
                    }
                };
                match self.tables.expr_ty_adjusted(&hir_node).sty {
                    ty::Adt(def, _) if !def.is_enum() => {
                        let variant = &def.non_enum_variant();
                        let index = self.tcx.find_field_index(ident, variant).unwrap();
                        filter!(self.span_utils, ident.span);
                        let span = self.span_from_span(ident.span);
                        return Some(Data::RefData(Ref {
                            kind: RefKind::Variable,
                            span,
                            ref_id: id_from_def_id(variant.fields[index].did),
                        }));
                    }
                    ty::Tuple(..) => None,
                    _ => {
                        debug!("Expected struct or union type, found {:?}", ty);
                        None
                    }
                }
            }
            ast::ExprKind::Struct(ref path, ..) => {
                match self.tables.expr_ty_adjusted(&hir_node).sty {
                    ty::Adt(def, _) if !def.is_enum() => {
                        let sub_span = path.segments.last().unwrap().ident.span;
                        filter!(self.span_utils, sub_span);
                        let span = self.span_from_span(sub_span);
                        Some(Data::RefData(Ref {
                            kind: RefKind::Type,
                            span,
                            ref_id: id_from_def_id(def.did),
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
            ast::ExprKind::MethodCall(ref seg, ..) => {
                let expr_hir_id = self.tcx.hir().definitions().node_to_hir_id(expr.id);
                let method_id = match self.tables.type_dependent_def_id(expr_hir_id) {
                    Some(id) => id,
                    None => {
                        debug!("Could not resolve method id for {:?}", expr);
                        return None;
                    }
                };
                let (def_id, decl_id) = match self.tcx.associated_item(method_id).container {
                    ty::ImplContainer(_) => (Some(method_id), None),
                    ty::TraitContainer(_) => (None, Some(method_id)),
                };
                let sub_span = seg.ident.span;
                filter!(self.span_utils, sub_span);
                let span = self.span_from_span(sub_span);
                Some(Data::RefData(Ref {
                    kind: RefKind::Function,
                    span,
                    ref_id: def_id
                        .or(decl_id)
                        .map(|id| id_from_def_id(id))
                        .unwrap_or_else(|| null_id()),
                }))
            }
            ast::ExprKind::Path(_, ref path) => {
                self.get_path_data(expr.id, path).map(|d| Data::RefData(d))
            }
            _ => {
                // FIXME
                bug!();
            }
        }
    }

    pub fn get_path_res(&self, id: NodeId) -> Res {
        let hir_id = self.tcx.hir().node_to_hir_id(id);
        match self.tcx.hir().get(hir_id) {
            Node::TraitRef(tr) => tr.path.res,

            Node::Item(&hir::Item {
                node: hir::ItemKind::Use(ref path, _),
                ..
            }) |
            Node::Visibility(&Spanned {
                node: hir::VisibilityKind::Restricted { ref path, .. }, .. }) => path.res,

            Node::PathSegment(seg) => {
                match seg.res {
                    Some(res) if res != Res::Err => res,
                    _ => {
                        let parent_node = self.tcx.hir().get_parent_node(hir_id);
                        self.get_path_res(self.tcx.hir().hir_to_node_id(parent_node))
                    },
                }
            }

            Node::Expr(&hir::Expr {
                node: hir::ExprKind::Struct(ref qpath, ..),
                ..
            }) => {
                self.tables.qpath_res(qpath, hir_id)
            }

            Node::Expr(&hir::Expr {
                node: hir::ExprKind::Path(ref qpath),
                ..
            }) |
            Node::Pat(&hir::Pat {
                node: hir::PatKind::Path(ref qpath),
                ..
            }) |
            Node::Pat(&hir::Pat {
                node: hir::PatKind::Struct(ref qpath, ..),
                ..
            }) |
            Node::Pat(&hir::Pat {
                node: hir::PatKind::TupleStruct(ref qpath, ..),
                ..
            }) |
            Node::Ty(&hir::Ty {
                node: hir::TyKind::Path(ref qpath),
                ..
            }) => {
                self.tables.qpath_res(qpath, hir_id)
            }

            Node::Binding(&hir::Pat {
                node: hir::PatKind::Binding(_, canonical_id, ..),
                ..
            }) => Res::Local(canonical_id),

            _ => Res::Err,
        }
    }

    pub fn get_path_data(&self, id: NodeId, path: &ast::Path) -> Option<Ref> {
        path.segments
            .last()
            .and_then(|seg| {
                self.get_path_segment_data(seg)
                    .or_else(|| self.get_path_segment_data_with_id(seg, id))
            })
    }

    pub fn get_path_segment_data(&self, path_seg: &ast::PathSegment) -> Option<Ref> {
        self.get_path_segment_data_with_id(path_seg, path_seg.id)
    }

    fn get_path_segment_data_with_id(
        &self,
        path_seg: &ast::PathSegment,
        id: NodeId,
    ) -> Option<Ref> {
        // Returns true if the path is function type sugar, e.g., `Fn(A) -> B`.
        fn fn_type(seg: &ast::PathSegment) -> bool {
            if let Some(ref generic_args) = seg.args {
                if let ast::GenericArgs::Parenthesized(_) = **generic_args {
                    return true;
                }
            }
            false
        }

        if id == DUMMY_NODE_ID {
            return None;
        }

        let res = self.get_path_res(id);
        let span = path_seg.ident.span;
        filter!(self.span_utils, span);
        let span = self.span_from_span(span);

        match res {
            Res::Local(id) => {
                Some(Ref {
                    kind: RefKind::Variable,
                    span,
                    ref_id: id_from_node_id(self.tcx.hir().hir_to_node_id(id), self),
                })
            }
            Res::Def(HirDefKind::Trait, def_id) if fn_type(path_seg) => {
                Some(Ref {
                    kind: RefKind::Type,
                    span,
                    ref_id: id_from_def_id(def_id),
                })
            }
            Res::Def(HirDefKind::Struct, def_id) |
            Res::Def(HirDefKind::Variant, def_id) |
            Res::Def(HirDefKind::Union, def_id) |
            Res::Def(HirDefKind::Enum, def_id) |
            Res::Def(HirDefKind::TyAlias, def_id) |
            Res::Def(HirDefKind::ForeignTy, def_id) |
            Res::Def(HirDefKind::TraitAlias, def_id) |
            Res::Def(HirDefKind::AssocExistential, def_id) |
            Res::Def(HirDefKind::AssocTy, def_id) |
            Res::Def(HirDefKind::Trait, def_id) |
            Res::Def(HirDefKind::Existential, def_id) |
            Res::Def(HirDefKind::TyParam, def_id) => {
                Some(Ref {
                    kind: RefKind::Type,
                    span,
                    ref_id: id_from_def_id(def_id),
                })
            }
            Res::Def(HirDefKind::ConstParam, def_id) => {
                Some(Ref {
                    kind: RefKind::Variable,
                    span,
                    ref_id: id_from_def_id(def_id),
                })
            }
            Res::Def(HirDefKind::Ctor(CtorOf::Struct, ..), def_id) => {
                // This is a reference to a tuple struct where the def_id points
                // to an invisible constructor function. That is not a very useful
                // def, so adjust to point to the tuple struct itself.
                let parent_def_id = self.tcx.parent(def_id).unwrap();
                Some(Ref {
                    kind: RefKind::Type,
                    span,
                    ref_id: id_from_def_id(parent_def_id),
                })
            }
            Res::Def(HirDefKind::Static, _) |
            Res::Def(HirDefKind::Const, _) |
            Res::Def(HirDefKind::AssocConst, _) |
            Res::Def(HirDefKind::Ctor(..), _) => {
                Some(Ref {
                    kind: RefKind::Variable,
                    span,
                    ref_id: id_from_def_id(res.def_id()),
                })
            }
            Res::Def(HirDefKind::Method, decl_id) => {
                let def_id = if decl_id.is_local() {
                    let ti = self.tcx.associated_item(decl_id);
                    self.tcx
                        .associated_items(ti.container.id())
                        .find(|item| item.ident.name == ti.ident.name &&
                                     item.defaultness.has_value())
                        .map(|item| item.def_id)
                } else {
                    None
                };
                Some(Ref {
                    kind: RefKind::Function,
                    span,
                    ref_id: id_from_def_id(def_id.unwrap_or(decl_id)),
                })
            }
            Res::Def(HirDefKind::Fn, def_id) => {
                Some(Ref {
                    kind: RefKind::Function,
                    span,
                    ref_id: id_from_def_id(def_id),
                })
            }
            Res::Def(HirDefKind::Mod, def_id) => {
                Some(Ref {
                    kind: RefKind::Mod,
                    span,
                    ref_id: id_from_def_id(def_id),
                })
            }
            Res::PrimTy(..) |
            Res::SelfTy(..) |
            Res::Def(HirDefKind::Macro(..), _) |
            Res::ToolMod |
            Res::NonMacroAttr(..) |
            Res::SelfCtor(..) |
            Res::Err => None,
        }
    }

    pub fn get_field_ref_data(
        &self,
        field_ref: &ast::Field,
        variant: &ty::VariantDef,
    ) -> Option<Ref> {
        filter!(self.span_utils, field_ref.ident.span);
        self.tcx.find_field_index(field_ref.ident, variant).map(|index| {
            let span = self.span_from_span(field_ref.ident.span);
            Ref {
                kind: RefKind::Variable,
                span,
                ref_id: id_from_def_id(variant.fields[index].did),
            }
        })
    }

    /// Attempt to return MacroRef for any AST node.
    ///
    /// For a given piece of AST defined by the supplied Span and NodeId,
    /// returns `None` if the node is not macro-generated or the span is malformed,
    /// else uses the expansion callsite and callee to return some MacroRef.
    pub fn get_macro_use_data(&self, span: Span) -> Option<MacroRef> {
        if !generated_code(span) {
            return None;
        }
        // Note we take care to use the source callsite/callee, to handle
        // nested expansions and ensure we only generate data for source-visible
        // macro uses.
        let callsite = span.source_callsite();
        let callsite_span = self.span_from_span(callsite);
        let callee = span.source_callee()?;
        let callee_span = callee.def_site?;

        // Ignore attribute macros, their spans are usually mangled
        if let MacroAttribute(_) = callee.format {
            return None;
        }

        // If the callee is an imported macro from an external crate, need to get
        // the source span and name from the session, as their spans are localized
        // when read in, and no longer correspond to the source.
        if let Some(mac) = self.tcx
            .sess
            .imported_macro_spans
            .borrow()
            .get(&callee_span)
        {
            let &(ref mac_name, mac_span) = mac;
            let mac_span = self.span_from_span(mac_span);
            return Some(MacroRef {
                span: callsite_span,
                qualname: mac_name.clone(), // FIXME: generate the real qualname
                callee_span: mac_span,
            });
        }

        let callee_span = self.span_from_span(callee_span);
        Some(MacroRef {
            span: callsite_span,
            qualname: callee.format.name().to_string(), // FIXME: generate the real qualname
            callee_span,
        })
    }

    fn lookup_ref_id(&self, ref_id: NodeId) -> Option<DefId> {
        match self.get_path_res(ref_id) {
            Res::PrimTy(_) | Res::SelfTy(..) | Res::Err => None,
            def => Some(def.def_id()),
        }
    }

    fn docs_for_attrs(&self, attrs: &[Attribute]) -> String {
        let mut result = String::new();

        for attr in attrs {
            if attr.check_name(sym::doc) {
                if let Some(val) = attr.value_str() {
                    if attr.is_sugared_doc {
                        result.push_str(&strip_doc_comment_decoration(&val.as_str()));
                    } else {
                        result.push_str(&val.as_str());
                    }
                    result.push('\n');
                } else if let Some(meta_list) = attr.meta_item_list() {
                    meta_list.into_iter()
                             .filter(|it| it.check_name(sym::include))
                             .filter_map(|it| it.meta_item_list().map(|l| l.to_owned()))
                             .flat_map(|it| it)
                             .filter(|meta| meta.check_name(sym::contents))
                             .filter_map(|meta| meta.value_str())
                             .for_each(|val| {
                                 result.push_str(&val.as_str());
                                 result.push('\n');
                             });
                }
            }
        }

        if !self.config.full_docs {
            if let Some(index) = result.find("\n\n") {
                result.truncate(index);
            }
        }

        result
    }

    fn next_impl_id(&self) -> u32 {
        let next = self.impl_counter.get();
        self.impl_counter.set(next + 1);
        next
    }
}

fn make_signature(decl: &ast::FnDecl, generics: &ast::Generics) -> String {
    let mut sig = "fn ".to_owned();
    if !generics.params.is_empty() {
        sig.push('<');
        sig.push_str(&generics
            .params
            .iter()
            .map(|param| param.ident.to_string())
            .collect::<Vec<_>>()
            .join(", "));
        sig.push_str("> ");
    }
    sig.push('(');
    sig.push_str(&decl.inputs
        .iter()
        .map(arg_to_string)
        .collect::<Vec<_>>()
        .join(", "));
    sig.push(')');
    match decl.output {
        ast::FunctionRetTy::Default(_) => sig.push_str(" -> ()"),
        ast::FunctionRetTy::Ty(ref t) => sig.push_str(&format!(" -> {}", ty_to_string(t))),
    }

    sig
}

// An AST visitor for collecting paths (e.g., the names of structs) and formal
// variables (idents) from patterns.
struct PathCollector<'l> {
    collected_paths: Vec<(NodeId, &'l ast::Path)>,
    collected_idents: Vec<(NodeId, ast::Ident, ast::Mutability)>,
}

impl<'l> PathCollector<'l> {
    fn new() -> PathCollector<'l> {
        PathCollector {
            collected_paths: vec![],
            collected_idents: vec![],
        }
    }
}

impl<'l> Visitor<'l> for PathCollector<'l> {
    fn visit_pat(&mut self, p: &'l ast::Pat) {
        match p.node {
            PatKind::Struct(ref path, ..) => {
                self.collected_paths.push((p.id, path));
            }
            PatKind::TupleStruct(ref path, ..) | PatKind::Path(_, ref path) => {
                self.collected_paths.push((p.id, path));
            }
            PatKind::Ident(bm, ident, _) => {
                debug!(
                    "PathCollector, visit ident in pat {}: {:?} {:?}",
                    ident,
                    p.span,
                    ident.span
                );
                let immut = match bm {
                    // Even if the ref is mut, you can't change the ref, only
                    // the data pointed at, so showing the initialising expression
                    // is still worthwhile.
                    ast::BindingMode::ByRef(_) => ast::Mutability::Immutable,
                    ast::BindingMode::ByValue(mt) => mt,
                };
                self.collected_idents
                    .push((p.id, ident, immut));
            }
            _ => {}
        }
        visit::walk_pat(self, p);
    }
}

/// Defines what to do with the results of saving the analysis.
pub trait SaveHandler {
    fn save<'l, 'tcx>(
        &mut self,
        save_ctxt: SaveContext<'l, 'tcx>,
        krate: &ast::Crate,
        cratename: &str,
        input: &'l Input,
    );
}

/// Dump the save-analysis results to a file.
pub struct DumpHandler<'a> {
    odir: Option<&'a Path>,
    cratename: String,
}

impl<'a> DumpHandler<'a> {
    pub fn new(odir: Option<&'a Path>, cratename: &str) -> DumpHandler<'a> {
        DumpHandler {
            odir,
            cratename: cratename.to_owned(),
        }
    }

    fn output_file(&self, ctx: &SaveContext<'_, '_>) -> (BufWriter<File>, PathBuf) {
        let sess = &ctx.tcx.sess;
        let file_name = match ctx.config.output_file {
            Some(ref s) => PathBuf::from(s),
            None => {
                let mut root_path = match self.odir {
                    Some(val) => val.join("save-analysis"),
                    None => PathBuf::from("save-analysis-temp"),
                };

                if let Err(e) = std::fs::create_dir_all(&root_path) {
                    error!("Could not create directory {}: {}", root_path.display(), e);
                }

                let executable = sess.crate_types
                    .borrow()
                    .iter()
                    .any(|ct| *ct == CrateType::Executable);
                let mut out_name = if executable {
                    String::new()
                } else {
                    "lib".to_owned()
                };
                out_name.push_str(&self.cratename);
                out_name.push_str(&sess.opts.cg.extra_filename);
                out_name.push_str(".json");
                root_path.push(&out_name);

                root_path
            }
        };

        info!("Writing output to {}", file_name.display());

        let output_file = BufWriter::new(File::create(&file_name).unwrap_or_else(
            |e| sess.fatal(&format!("Could not open {}: {}", file_name.display(), e)),
        ));

        (output_file, file_name)
    }
}

impl<'a> SaveHandler for DumpHandler<'a> {
    fn save<'l, 'tcx>(
        &mut self,
        save_ctxt: SaveContext<'l, 'tcx>,
        krate: &ast::Crate,
        cratename: &str,
        input: &'l Input,
    ) {
        let sess = &save_ctxt.tcx.sess;
        let file_name = {
            let (mut output, file_name) = self.output_file(&save_ctxt);
            let mut dumper = JsonDumper::new(&mut output, save_ctxt.config.clone());
            let mut visitor = DumpVisitor::new(save_ctxt, &mut dumper);

            visitor.dump_crate_info(cratename, krate);
            visitor.dump_compilation_options(input, cratename);
            visit::walk_crate(&mut visitor, krate);

            file_name
        };

        if sess.opts.debugging_opts.emit_artifact_notifications {
            sess.parse_sess.span_diagnostic
                .emit_artifact_notification(&file_name, "save-analysis");
        }
    }
}

/// Call a callback with the results of save-analysis.
pub struct CallbackHandler<'b> {
    pub callback: &'b mut dyn FnMut(&rls_data::Analysis),
}

impl<'b> SaveHandler for CallbackHandler<'b> {
    fn save<'l, 'tcx>(
        &mut self,
        save_ctxt: SaveContext<'l, 'tcx>,
        krate: &ast::Crate,
        cratename: &str,
        input: &'l Input,
    ) {
        // We're using the JsonDumper here because it has the format of the
        // save-analysis results that we will pass to the callback. IOW, we are
        // using the JsonDumper to collect the save-analysis results, but not
        // actually to dump them to a file. This is all a bit convoluted and
        // there is certainly a simpler design here trying to get out (FIXME).
        let mut dumper = JsonDumper::with_callback(self.callback, save_ctxt.config.clone());
        let mut visitor = DumpVisitor::new(save_ctxt, &mut dumper);

        visitor.dump_crate_info(cratename, krate);
        visitor.dump_compilation_options(input, cratename);
        visit::walk_crate(&mut visitor, krate);
    }
}

pub fn process_crate<'l, 'tcx, H: SaveHandler>(
    tcx: TyCtxt<'tcx>,
    krate: &ast::Crate,
    cratename: &str,
    input: &'l Input,
    config: Option<Config>,
    mut handler: H,
) {
    tcx.dep_graph.with_ignore(|| {
        info!("Dumping crate {}", cratename);

        // Privacy checking requires and is done after type checking; use a
        // fallback in case the access levels couldn't have been correctly computed.
        let access_levels = match tcx.sess.compile_status() {
            Ok(..) => tcx.privacy_access_levels(LOCAL_CRATE),
            Err(..) => tcx.arena.alloc(AccessLevels::default()),
        };

        let save_ctxt = SaveContext {
            tcx,
            tables: &ty::TypeckTables::empty(None),
            access_levels: &access_levels,
            span_utils: SpanUtils::new(&tcx.sess),
            config: find_config(config),
            impl_counter: Cell::new(0),
        };

        handler.save(save_ctxt, krate, cratename, input)
    })
}

fn find_config(supplied: Option<Config>) -> Config {
    if let Some(config) = supplied {
        return config;
    }

    match env::var_os("RUST_SAVE_ANALYSIS_CONFIG") {
        None => Config::default(),
        Some(config) => config.to_str()
            .ok_or(())
            .map_err(|_| error!("`RUST_SAVE_ANALYSIS_CONFIG` isn't UTF-8"))
            .and_then(|cfg|  serde_json::from_str(cfg)
                .map_err(|_| error!("Could not deserialize save-analysis config"))
            ).unwrap_or_default()
    }
}

// Utility functions for the module.

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

// Helper function to determine if a span came from a
// macro expansion or syntax extension.
fn generated_code(span: Span) -> bool {
    span.ctxt() != NO_EXPANSION || span.is_dummy()
}

// DefId::index is a newtype and so the JSON serialisation is ugly. Therefore
// we use our own Id which is the same, but without the newtype.
fn id_from_def_id(id: DefId) -> rls_data::Id {
    rls_data::Id {
        krate: id.krate.as_u32(),
        index: id.index.as_u32(),
    }
}

fn id_from_node_id(id: NodeId, scx: &SaveContext<'_, '_>) -> rls_data::Id {
    let def_id = scx.tcx.hir().opt_local_def_id(id);
    def_id.map(|id| id_from_def_id(id)).unwrap_or_else(|| {
        // Create a *fake* `DefId` out of a `NodeId` by subtracting the `NodeId`
        // out of the maximum u32 value. This will work unless you have *billions*
        // of definitions in a single crate (very unlikely to actually happen).
        rls_data::Id {
            krate: LOCAL_CRATE.as_u32(),
            index: !id.as_u32(),
        }
    })
}

fn null_id() -> rls_data::Id {
    rls_data::Id {
        krate: u32::max_value(),
        index: u32::max_value(),
    }
}

fn lower_attributes(attrs: Vec<Attribute>, scx: &SaveContext<'_, '_>) -> Vec<rls_data::Attribute> {
    attrs.into_iter()
    // Only retain real attributes. Doc comments are lowered separately.
    .filter(|attr| attr.path != sym::doc)
    .map(|mut attr| {
        // Remove the surrounding '#[..]' or '#![..]' of the pretty printed
        // attribute. First normalize all inner attribute (#![..]) to outer
        // ones (#[..]), then remove the two leading and the one trailing character.
        attr.style = ast::AttrStyle::Outer;
        let value = pprust::attribute_to_string(&attr);
        // This str slicing works correctly, because the leading and trailing characters
        // are in the ASCII range and thus exactly one byte each.
        let value = value[2..value.len()-1].to_string();

        rls_data::Attribute {
            value,
            span: scx.span_from_span(attr.span),
        }
    }).collect()
}
