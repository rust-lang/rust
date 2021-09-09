#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(if_let_guard)]
#![feature(nll)]
#![recursion_limit = "256"]

mod dump_visitor;
mod dumper;
#[macro_use]
mod span_utils;
mod sig;

use rustc_ast as ast;
use rustc_ast::util::comments::beautify_doc_string;
use rustc_ast_pretty::pprust::attribute_to_string;
use rustc_hir as hir;
use rustc_hir::def::{DefKind as HirDefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::Node;
use rustc_hir_pretty::{enum_def_to_string, fn_to_string, ty_to_string};
use rustc_middle::hir::map::Map;
use rustc_middle::middle::cstore::ExternCrate;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::ty::{self, print::with_no_trimmed_paths, DefIdTree, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::config::{CrateType, Input, OutputType};
use rustc_session::output::{filename_for_metadata, out_filename};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::*;

use std::cell::Cell;
use std::default::Default;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dump_visitor::DumpVisitor;
use span_utils::SpanUtils;

use rls_data::config::Config;
use rls_data::{
    Analysis, Def, DefKind, ExternalCrateData, GlobalCrateId, Impl, ImplKind, MacroRef, Ref,
    RefKind, Relation, RelationKind, SpanData,
};

use tracing::{debug, error, info};

pub struct SaveContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    access_levels: &'tcx AccessLevels,
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

impl<'tcx> SaveContext<'tcx> {
    /// Gets the type-checking results for the current body.
    /// As this will ICE if called outside bodies, only call when working with
    /// `Expr` or `Pat` nodes (they are guaranteed to be found only in bodies).
    #[track_caller]
    fn typeck_results(&self) -> &'tcx ty::TypeckResults<'tcx> {
        self.maybe_typeck_results.expect("`SaveContext::typeck_results` called outside of body")
    }

    fn span_from_span(&self, span: Span) -> SpanData {
        use rls_span::{Column, Row};

        let sm = self.tcx.sess.source_map();
        let start = sm.lookup_char_pos(span.lo());
        let end = sm.lookup_char_pos(span.hi());

        SpanData {
            file_name: start.file.name.prefer_remapped().to_string().into(),
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
        let crate_type = sess.crate_types()[0];
        let outputs = &*self.tcx.output_filenames(());

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
        let mut result = Vec::with_capacity(self.tcx.crates(()).len());

        for &n in self.tcx.crates(()).iter() {
            let span = match self.tcx.extern_crate(n.as_def_id()) {
                Some(&ExternCrate { span, .. }) => span,
                None => {
                    debug!("skipping crate {}, no data", n);
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
                    disambiguator: (
                        self.tcx.def_path_hash(n.as_def_id()).stable_crate_id().to_u64(),
                        0,
                    ),
                },
            });
        }

        result
    }

    pub fn get_extern_item_data(&self, item: &hir::ForeignItem<'_>) -> Option<Data> {
        let def_id = item.def_id.to_def_id();
        let qualname = format!("::{}", self.tcx.def_path_str(def_id));
        let attrs = self.tcx.hir().attrs(item.hir_id());
        match item.kind {
            hir::ForeignItemKind::Fn(ref decl, arg_names, ref generics) => {
                filter!(self.span_utils, item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::ForeignFunction,
                    id: id_from_def_id(def_id),
                    span: self.span_from_span(item.ident.span),
                    name: item.ident.to_string(),
                    qualname,
                    value: fn_to_string(
                        decl,
                        hir::FnHeader {
                            // functions in extern block are implicitly unsafe
                            unsafety: hir::Unsafety::Unsafe,
                            // functions in extern block cannot be const
                            constness: hir::Constness::NotConst,
                            abi: self.tcx.hir().get_foreign_abi(item.hir_id()),
                            // functions in extern block cannot be async
                            asyncness: hir::IsAsync::NotAsync,
                        },
                        Some(item.ident.name),
                        generics,
                        &item.vis,
                        arg_names,
                        None,
                    ),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::foreign_item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ForeignItemKind::Static(ref ty, _) => {
                filter!(self.span_utils, item.ident.span);

                let id = id_from_def_id(def_id);
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
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::foreign_item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            // FIXME(plietar): needs a new DefKind in rls-data
            hir::ForeignItemKind::Type => None,
        }
    }

    pub fn get_item_data(&self, item: &hir::Item<'_>) -> Option<Data> {
        let def_id = item.def_id.to_def_id();
        let attrs = self.tcx.hir().attrs(item.hir_id());
        match item.kind {
            hir::ItemKind::Fn(ref sig, ref generics, _) => {
                let qualname = format!("::{}", self.tcx.def_path_str(def_id));
                filter!(self.span_utils, item.ident.span);
                Some(Data::DefData(Def {
                    kind: DefKind::Function,
                    id: id_from_def_id(def_id),
                    span: self.span_from_span(item.ident.span),
                    name: item.ident.to_string(),
                    qualname,
                    value: fn_to_string(
                        sig.decl,
                        sig.header,
                        Some(item.ident.name),
                        generics,
                        &item.vis,
                        &[],
                        None,
                    ),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ItemKind::Static(ref typ, ..) => {
                let qualname = format!("::{}", self.tcx.def_path_str(def_id));

                filter!(self.span_utils, item.ident.span);

                let id = id_from_def_id(def_id);
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
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ItemKind::Const(ref typ, _) => {
                let qualname = format!("::{}", self.tcx.def_path_str(def_id));
                filter!(self.span_utils, item.ident.span);

                let id = id_from_def_id(def_id);
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
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ItemKind::Mod(ref m) => {
                let qualname = format!("::{}", self.tcx.def_path_str(def_id));

                let sm = self.tcx.sess.source_map();
                let filename = sm.span_to_filename(m.inner);

                filter!(self.span_utils, item.ident.span);

                Some(Data::DefData(Def {
                    kind: DefKind::Mod,
                    id: id_from_def_id(def_id),
                    name: item.ident.to_string(),
                    qualname,
                    span: self.span_from_span(item.ident.span),
                    value: filename.prefer_remapped().to_string(),
                    parent: None,
                    children: m
                        .item_ids
                        .iter()
                        .map(|i| id_from_def_id(i.def_id.to_def_id()))
                        .collect(),
                    decl_id: None,
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ItemKind::Enum(ref def, ref generics) => {
                let name = item.ident.to_string();
                let qualname = format!("::{}", self.tcx.def_path_str(def_id));
                filter!(self.span_utils, item.ident.span);
                let value =
                    enum_def_to_string(def, generics, item.ident.name, item.span, &item.vis);
                Some(Data::DefData(Def {
                    kind: DefKind::Enum,
                    id: id_from_def_id(def_id),
                    span: self.span_from_span(item.ident.span),
                    name,
                    qualname,
                    value,
                    parent: None,
                    children: def.variants.iter().map(|v| id_from_hir_id(v.id, self)).collect(),
                    decl_id: None,
                    docs: self.docs_for_attrs(attrs),
                    sig: sig::item_signature(item, self),
                    attributes: lower_attributes(attrs.to_vec(), self),
                }))
            }
            hir::ItemKind::Impl(hir::Impl { ref of_trait, ref self_ty, ref items, .. })
                if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = self_ty.kind =>
            {
                // Common case impl for a struct or something basic.
                if generated_code(path.span) {
                    return None;
                }
                let sub_span = path.segments.last().unwrap().ident.span;
                filter!(self.span_utils, sub_span);

                let impl_id = self.next_impl_id();
                let span = self.span_from_span(sub_span);

                let type_data = self.lookup_def_id(self_ty.hir_id);
                type_data.map(|type_data| {
                    Data::RelationData(
                        Relation {
                            kind: RelationKind::Impl { id: impl_id },
                            span: span.clone(),
                            from: id_from_def_id(type_data),
                            to: of_trait
                                .as_ref()
                                .and_then(|t| self.lookup_def_id(t.hir_ref_id))
                                .map(id_from_def_id)
                                .unwrap_or_else(null_id),
                        },
                        Impl {
                            id: impl_id,
                            kind: match *of_trait {
                                Some(_) => ImplKind::Direct,
                                None => ImplKind::Inherent,
                            },
                            span,
                            value: String::new(),
                            parent: None,
                            children: items
                                .iter()
                                .map(|i| id_from_def_id(i.id.def_id.to_def_id()))
                                .collect(),
                            docs: String::new(),
                            sig: None,
                            attributes: vec![],
                        },
                    )
                })
            }
            hir::ItemKind::Impl(_) => None,
            _ => {
                // FIXME
                bug!();
            }
        }
    }

    pub fn get_field_data(&self, field: &hir::FieldDef<'_>, scope: hir::HirId) -> Option<Def> {
        let name = field.ident.to_string();
        let scope_def_id = self.tcx.hir().local_def_id(scope).to_def_id();
        let qualname = format!("::{}::{}", self.tcx.def_path_str(scope_def_id), field.ident);
        filter!(self.span_utils, field.ident.span);
        let field_def_id = self.tcx.hir().local_def_id(field.hir_id).to_def_id();
        let typ = self.tcx.type_of(field_def_id).to_string();

        let id = id_from_def_id(field_def_id);
        let span = self.span_from_span(field.ident.span);
        let attrs = self.tcx.hir().attrs(field.hir_id);

        Some(Def {
            kind: DefKind::Field,
            id,
            span,
            name,
            qualname,
            value: typ,
            parent: Some(id_from_def_id(scope_def_id)),
            children: vec![],
            decl_id: None,
            docs: self.docs_for_attrs(attrs),
            sig: sig::field_signature(field, self),
            attributes: lower_attributes(attrs.to_vec(), self),
        })
    }

    // FIXME would be nice to take a MethodItem here, but the ast provides both
    // trait and impl flavours, so the caller must do the disassembly.
    pub fn get_method_data(&self, hir_id: hir::HirId, ident: Ident, span: Span) -> Option<Def> {
        // The qualname for a method is the trait name or name of the struct in an impl in
        // which the method is declared in, followed by the method's name.
        let def_id = self.tcx.hir().local_def_id(hir_id).to_def_id();
        let (qualname, parent_scope, decl_id, docs, attributes) =
            match self.tcx.impl_of_method(def_id) {
                Some(impl_id) => match self.tcx.hir().get_if_local(impl_id) {
                    Some(Node::Item(item)) => match item.kind {
                        hir::ItemKind::Impl(hir::Impl { ref self_ty, .. }) => {
                            let hir = self.tcx.hir();

                            let mut qualname = String::from("<");
                            qualname
                                .push_str(&rustc_hir_pretty::id_to_string(&hir, self_ty.hir_id));

                            let trait_id = self.tcx.trait_id_of_impl(impl_id);
                            let mut docs = String::new();
                            let mut attrs = vec![];
                            if let Some(Node::ImplItem(_)) = hir.find(hir_id) {
                                attrs = self.tcx.hir().attrs(hir_id).to_vec();
                                docs = self.docs_for_attrs(&attrs);
                            }

                            let mut decl_id = None;
                            if let Some(def_id) = trait_id {
                                // A method in a trait impl.
                                qualname.push_str(" as ");
                                qualname.push_str(&self.tcx.def_path_str(def_id));

                                decl_id = self
                                    .tcx
                                    .associated_items(def_id)
                                    .filter_by_name_unhygienic(ident.name)
                                    .next()
                                    .map(|item| item.def_id);
                            }
                            qualname.push('>');

                            (qualname, trait_id, decl_id, docs, attrs)
                        }
                        _ => {
                            span_bug!(
                                span,
                                "Container {:?} for method {} not an impl?",
                                impl_id,
                                hir_id
                            );
                        }
                    },
                    r => {
                        span_bug!(
                            span,
                            "Container {:?} for method {} is not a node item {:?}",
                            impl_id,
                            hir_id,
                            r
                        );
                    }
                },
                None => match self.tcx.trait_of_item(def_id) {
                    Some(def_id) => {
                        let mut docs = String::new();
                        let mut attrs = vec![];

                        if let Some(Node::TraitItem(_)) = self.tcx.hir().find(hir_id) {
                            attrs = self.tcx.hir().attrs(hir_id).to_vec();
                            docs = self.docs_for_attrs(&attrs);
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
                        debug!("could not find container for method {} at {:?}", hir_id, span);
                        // This is not necessarily a bug, if there was a compilation error,
                        // the typeck results we need might not exist.
                        return None;
                    }
                },
            };

        let qualname = format!("{}::{}", qualname, ident.name);

        filter!(self.span_utils, ident.span);

        Some(Def {
            kind: DefKind::Method,
            id: id_from_def_id(def_id),
            span: self.span_from_span(ident.span),
            name: ident.name.to_string(),
            qualname,
            // FIXME you get better data here by using the visitor.
            value: String::new(),
            parent: parent_scope.map(id_from_def_id),
            children: vec![],
            decl_id: decl_id.map(id_from_def_id),
            docs,
            sig: None,
            attributes: lower_attributes(attributes, self),
        })
    }

    pub fn get_expr_data(&self, expr: &hir::Expr<'_>) -> Option<Data> {
        let ty = self.typeck_results().expr_ty_adjusted_opt(expr)?;
        if matches!(ty.kind(), ty::Error(_)) {
            return None;
        }
        match expr.kind {
            hir::ExprKind::Field(ref sub_ex, ident) => {
                match self.typeck_results().expr_ty_adjusted(&sub_ex).kind() {
                    ty::Adt(def, _) if !def.is_enum() => {
                        let variant = &def.non_enum_variant();
                        filter!(self.span_utils, ident.span);
                        let span = self.span_from_span(ident.span);
                        Some(Data::RefData(Ref {
                            kind: RefKind::Variable,
                            span,
                            ref_id: self
                                .tcx
                                .find_field_index(ident, variant)
                                .map(|index| id_from_def_id(variant.fields[index].did))
                                .unwrap_or_else(null_id),
                        }))
                    }
                    ty::Tuple(..) => None,
                    _ => {
                        debug!("expected struct or union type, found {:?}", ty);
                        None
                    }
                }
            }
            hir::ExprKind::Struct(qpath, ..) => match ty.kind() {
                ty::Adt(def, _) => {
                    let sub_span = qpath.last_segment_span();
                    filter!(self.span_utils, sub_span);
                    let span = self.span_from_span(sub_span);
                    Some(Data::RefData(Ref {
                        kind: RefKind::Type,
                        span,
                        ref_id: id_from_def_id(def.did),
                    }))
                }
                _ => {
                    debug!("expected adt, found {:?}", ty);
                    None
                }
            },
            hir::ExprKind::MethodCall(ref seg, ..) => {
                let method_id = match self.typeck_results().type_dependent_def_id(expr.hir_id) {
                    Some(id) => id,
                    None => {
                        debug!("could not resolve method id for {:?}", expr);
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
                    ref_id: def_id.or(decl_id).map(id_from_def_id).unwrap_or_else(null_id),
                }))
            }
            hir::ExprKind::Path(ref path) => {
                self.get_path_data(expr.hir_id, path).map(Data::RefData)
            }
            _ => {
                // FIXME
                bug!("invalid expression: {:?}", expr);
            }
        }
    }

    pub fn get_path_res(&self, hir_id: hir::HirId) -> Res {
        match self.tcx.hir().get(hir_id) {
            Node::TraitRef(tr) => tr.path.res,

            Node::Item(&hir::Item { kind: hir::ItemKind::Use(path, _), .. }) => path.res,
            Node::Visibility(&Spanned {
                node: hir::VisibilityKind::Restricted { ref path, .. },
                ..
            }) => path.res,

            Node::PathSegment(seg) => match seg.res {
                Some(res) if res != Res::Err => res,
                _ => {
                    let parent_node = self.tcx.hir().get_parent_node(hir_id);
                    self.get_path_res(parent_node)
                }
            },

            Node::Expr(&hir::Expr { kind: hir::ExprKind::Struct(ref qpath, ..), .. }) => {
                self.typeck_results().qpath_res(qpath, hir_id)
            }

            Node::Expr(&hir::Expr { kind: hir::ExprKind::Path(ref qpath), .. })
            | Node::Pat(&hir::Pat {
                kind:
                    hir::PatKind::Path(ref qpath)
                    | hir::PatKind::Struct(ref qpath, ..)
                    | hir::PatKind::TupleStruct(ref qpath, ..),
                ..
            })
            | Node::Ty(&hir::Ty { kind: hir::TyKind::Path(ref qpath), .. }) => match qpath {
                hir::QPath::Resolved(_, path) => path.res,
                hir::QPath::TypeRelative(..) | hir::QPath::LangItem(..) => {
                    // #75962: `self.typeck_results` may be different from the `hir_id`'s result.
                    if self.tcx.has_typeck_results(hir_id.owner.to_def_id()) {
                        self.tcx.typeck(hir_id.owner).qpath_res(qpath, hir_id)
                    } else {
                        Res::Err
                    }
                }
            },

            Node::Binding(&hir::Pat {
                kind: hir::PatKind::Binding(_, canonical_id, ..), ..
            }) => Res::Local(canonical_id),

            _ => Res::Err,
        }
    }

    pub fn get_path_data(&self, id: hir::HirId, path: &hir::QPath<'_>) -> Option<Ref> {
        let segment = match path {
            hir::QPath::Resolved(_, path) => path.segments.last(),
            hir::QPath::TypeRelative(_, segment) => Some(*segment),
            hir::QPath::LangItem(..) => None,
        };
        segment.and_then(|seg| {
            self.get_path_segment_data(seg).or_else(|| self.get_path_segment_data_with_id(seg, id))
        })
    }

    pub fn get_path_segment_data(&self, path_seg: &hir::PathSegment<'_>) -> Option<Ref> {
        self.get_path_segment_data_with_id(path_seg, path_seg.hir_id?)
    }

    pub fn get_path_segment_data_with_id(
        &self,
        path_seg: &hir::PathSegment<'_>,
        id: hir::HirId,
    ) -> Option<Ref> {
        // Returns true if the path is function type sugar, e.g., `Fn(A) -> B`.
        fn fn_type(seg: &hir::PathSegment<'_>) -> bool {
            seg.args.map_or(false, |args| args.parenthesized)
        }

        let res = self.get_path_res(id);
        let span = path_seg.ident.span;
        filter!(self.span_utils, span);
        let span = self.span_from_span(span);

        match res {
            Res::Local(id) => {
                Some(Ref { kind: RefKind::Variable, span, ref_id: id_from_hir_id(id, self) })
            }
            Res::Def(HirDefKind::Trait, def_id) if fn_type(path_seg) => {
                Some(Ref { kind: RefKind::Type, span, ref_id: id_from_def_id(def_id) })
            }
            Res::Def(
                HirDefKind::Struct
                | HirDefKind::Variant
                | HirDefKind::Union
                | HirDefKind::Enum
                | HirDefKind::TyAlias
                | HirDefKind::ForeignTy
                | HirDefKind::TraitAlias
                | HirDefKind::AssocTy
                | HirDefKind::Trait
                | HirDefKind::OpaqueTy
                | HirDefKind::TyParam,
                def_id,
            ) => Some(Ref { kind: RefKind::Type, span, ref_id: id_from_def_id(def_id) }),
            Res::Def(HirDefKind::ConstParam, def_id) => {
                Some(Ref { kind: RefKind::Variable, span, ref_id: id_from_def_id(def_id) })
            }
            Res::Def(HirDefKind::Ctor(..), def_id) => {
                // This is a reference to a tuple struct or an enum variant where the def_id points
                // to an invisible constructor function. That is not a very useful
                // def, so adjust to point to the tuple struct or enum variant itself.
                let parent_def_id = self.tcx.parent(def_id).unwrap();
                Some(Ref { kind: RefKind::Type, span, ref_id: id_from_def_id(parent_def_id) })
            }
            Res::Def(HirDefKind::Static | HirDefKind::Const | HirDefKind::AssocConst, _) => {
                Some(Ref { kind: RefKind::Variable, span, ref_id: id_from_def_id(res.def_id()) })
            }
            Res::Def(HirDefKind::AssocFn, decl_id) => {
                let def_id = if decl_id.is_local() {
                    let ti = self.tcx.associated_item(decl_id);

                    self.tcx
                        .associated_items(ti.container.id())
                        .filter_by_name_unhygienic(ti.ident.name)
                        .find(|item| item.defaultness.has_value())
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
                Some(Ref { kind: RefKind::Function, span, ref_id: id_from_def_id(def_id) })
            }
            Res::Def(HirDefKind::Mod, def_id) => {
                Some(Ref { kind: RefKind::Mod, span, ref_id: id_from_def_id(def_id) })
            }

            Res::Def(
                HirDefKind::Macro(..)
                | HirDefKind::ExternCrate
                | HirDefKind::ForeignMod
                | HirDefKind::LifetimeParam
                | HirDefKind::AnonConst
                | HirDefKind::Use
                | HirDefKind::Field
                | HirDefKind::GlobalAsm
                | HirDefKind::Impl
                | HirDefKind::Closure
                | HirDefKind::Generator,
                _,
            )
            | Res::PrimTy(..)
            | Res::SelfTy(..)
            | Res::ToolMod
            | Res::NonMacroAttr(..)
            | Res::SelfCtor(..)
            | Res::Err => None,
        }
    }

    pub fn get_field_ref_data(
        &self,
        field_ref: &hir::ExprField<'_>,
        variant: &ty::VariantDef,
    ) -> Option<Ref> {
        filter!(self.span_utils, field_ref.ident.span);
        self.tcx.find_field_index(field_ref.ident, variant).map(|index| {
            let span = self.span_from_span(field_ref.ident.span);
            Ref { kind: RefKind::Variable, span, ref_id: id_from_def_id(variant.fields[index].did) }
        })
    }

    /// Attempt to return MacroRef for any AST node.
    ///
    /// For a given piece of AST defined by the supplied Span and NodeId,
    /// returns `None` if the node is not macro-generated or the span is malformed,
    /// else uses the expansion callsite and callee to return some MacroRef.
    ///
    /// FIXME: [`DumpVisitor::process_macro_use`] should actually dump this data
    #[allow(dead_code)]
    fn get_macro_use_data(&self, span: Span) -> Option<MacroRef> {
        if !generated_code(span) {
            return None;
        }
        // Note we take care to use the source callsite/callee, to handle
        // nested expansions and ensure we only generate data for source-visible
        // macro uses.
        let callsite = span.source_callsite();
        let callsite_span = self.span_from_span(callsite);
        let callee = span.source_callee()?;

        let mac_name = match callee.kind {
            ExpnKind::Macro(kind, name) => match kind {
                MacroKind::Bang => name,

                // Ignore attribute macros, their spans are usually mangled
                // FIXME(eddyb) is this really the case anymore?
                MacroKind::Attr | MacroKind::Derive => return None,
            },

            // These are not macros.
            // FIXME(eddyb) maybe there is a way to handle them usefully?
            ExpnKind::Inlined | ExpnKind::Root | ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => {
                return None;
            }
        };

        let callee_span = self.span_from_span(callee.def_site);
        Some(MacroRef {
            span: callsite_span,
            qualname: mac_name.to_string(), // FIXME: generate the real qualname
            callee_span,
        })
    }

    fn lookup_def_id(&self, ref_id: hir::HirId) -> Option<DefId> {
        match self.get_path_res(ref_id) {
            Res::PrimTy(_) | Res::SelfTy(..) | Res::Err => None,
            def => def.opt_def_id(),
        }
    }

    fn docs_for_attrs(&self, attrs: &[ast::Attribute]) -> String {
        let mut result = String::new();

        for attr in attrs {
            if let Some(val) = attr.doc_str() {
                // FIXME: Should save-analysis beautify doc strings itself or leave it to users?
                result.push_str(&beautify_doc_string(val).as_str());
                result.push('\n');
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

// An AST visitor for collecting paths (e.g., the names of structs) and formal
// variables (idents) from patterns.
struct PathCollector<'l> {
    tcx: TyCtxt<'l>,
    collected_paths: Vec<(hir::HirId, &'l hir::QPath<'l>)>,
    collected_idents: Vec<(hir::HirId, Ident, hir::Mutability)>,
}

impl<'l> PathCollector<'l> {
    fn new(tcx: TyCtxt<'l>) -> PathCollector<'l> {
        PathCollector { tcx, collected_paths: vec![], collected_idents: vec![] }
    }
}

impl<'l> Visitor<'l> for PathCollector<'l> {
    type Map = Map<'l>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_pat(&mut self, p: &'l hir::Pat<'l>) {
        match p.kind {
            hir::PatKind::Struct(ref path, ..) => {
                self.collected_paths.push((p.hir_id, path));
            }
            hir::PatKind::TupleStruct(ref path, ..) | hir::PatKind::Path(ref path) => {
                self.collected_paths.push((p.hir_id, path));
            }
            hir::PatKind::Binding(bm, _, ident, _) => {
                debug!(
                    "PathCollector, visit ident in pat {}: {:?} {:?}",
                    ident, p.span, ident.span
                );
                let immut = match bm {
                    // Even if the ref is mut, you can't change the ref, only
                    // the data pointed at, so showing the initialising expression
                    // is still worthwhile.
                    hir::BindingAnnotation::Unannotated | hir::BindingAnnotation::Ref => {
                        hir::Mutability::Not
                    }
                    hir::BindingAnnotation::Mutable | hir::BindingAnnotation::RefMut => {
                        hir::Mutability::Mut
                    }
                };
                self.collected_idents.push((p.hir_id, ident, immut));
            }
            _ => {}
        }
        intravisit::walk_pat(self, p);
    }
}

/// Defines what to do with the results of saving the analysis.
pub trait SaveHandler {
    fn save(&mut self, save_ctxt: &SaveContext<'_>, analysis: &Analysis);
}

/// Dump the save-analysis results to a file.
pub struct DumpHandler<'a> {
    odir: Option<&'a Path>,
    cratename: String,
}

impl<'a> DumpHandler<'a> {
    pub fn new(odir: Option<&'a Path>, cratename: &str) -> DumpHandler<'a> {
        DumpHandler { odir, cratename: cratename.to_owned() }
    }

    fn output_file(&self, ctx: &SaveContext<'_>) -> (BufWriter<File>, PathBuf) {
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

                let executable = sess.crate_types().iter().any(|ct| *ct == CrateType::Executable);
                let mut out_name = if executable { String::new() } else { "lib".to_owned() };
                out_name.push_str(&self.cratename);
                out_name.push_str(&sess.opts.cg.extra_filename);
                out_name.push_str(".json");
                root_path.push(&out_name);

                root_path
            }
        };

        info!("Writing output to {}", file_name.display());

        let output_file = BufWriter::new(File::create(&file_name).unwrap_or_else(|e| {
            sess.fatal(&format!("Could not open {}: {}", file_name.display(), e))
        }));

        (output_file, file_name)
    }
}

impl SaveHandler for DumpHandler<'_> {
    fn save(&mut self, save_ctxt: &SaveContext<'_>, analysis: &Analysis) {
        let sess = &save_ctxt.tcx.sess;
        let (output, file_name) = self.output_file(&save_ctxt);
        if let Err(e) = serde_json::to_writer(output, &analysis) {
            error!("Can't serialize save-analysis: {:?}", e);
        }

        if sess.opts.json_artifact_notifications {
            sess.parse_sess.span_diagnostic.emit_artifact_notification(&file_name, "save-analysis");
        }
    }
}

/// Call a callback with the results of save-analysis.
pub struct CallbackHandler<'b> {
    pub callback: &'b mut dyn FnMut(&rls_data::Analysis),
}

impl SaveHandler for CallbackHandler<'_> {
    fn save(&mut self, _: &SaveContext<'_>, analysis: &Analysis) {
        (self.callback)(analysis)
    }
}

pub fn process_crate<'l, 'tcx, H: SaveHandler>(
    tcx: TyCtxt<'tcx>,
    cratename: &str,
    input: &'l Input,
    config: Option<Config>,
    mut handler: H,
) {
    with_no_trimmed_paths(|| {
        tcx.dep_graph.with_ignore(|| {
            info!("Dumping crate {}", cratename);

            // Privacy checking requires and is done after type checking; use a
            // fallback in case the access levels couldn't have been correctly computed.
            let access_levels = match tcx.sess.compile_status() {
                Ok(..) => tcx.privacy_access_levels(()),
                Err(..) => tcx.arena.alloc(AccessLevels::default()),
            };

            let save_ctxt = SaveContext {
                tcx,
                maybe_typeck_results: None,
                access_levels: &access_levels,
                span_utils: SpanUtils::new(&tcx.sess),
                config: find_config(config),
                impl_counter: Cell::new(0),
            };

            let mut visitor = DumpVisitor::new(save_ctxt);

            visitor.dump_crate_info(cratename, tcx.hir().krate());
            visitor.dump_compilation_options(input, cratename);
            visitor.process_crate(tcx.hir().krate());

            handler.save(&visitor.save_ctxt, &visitor.analysis())
        })
    })
}

fn find_config(supplied: Option<Config>) -> Config {
    if let Some(config) = supplied {
        return config;
    }

    match env::var_os("RUST_SAVE_ANALYSIS_CONFIG") {
        None => Config::default(),
        Some(config) => config
            .to_str()
            .ok_or(())
            .map_err(|_| error!("`RUST_SAVE_ANALYSIS_CONFIG` isn't UTF-8"))
            .and_then(|cfg| {
                serde_json::from_str(cfg)
                    .map_err(|_| error!("Could not deserialize save-analysis config"))
            })
            .unwrap_or_default(),
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
    span.from_expansion() || span.is_dummy()
}

// DefId::index is a newtype and so the JSON serialisation is ugly. Therefore
// we use our own Id which is the same, but without the newtype.
fn id_from_def_id(id: DefId) -> rls_data::Id {
    rls_data::Id { krate: id.krate.as_u32(), index: id.index.as_u32() }
}

fn id_from_hir_id(id: hir::HirId, scx: &SaveContext<'_>) -> rls_data::Id {
    let def_id = scx.tcx.hir().opt_local_def_id(id);
    def_id.map(|id| id_from_def_id(id.to_def_id())).unwrap_or_else(|| {
        // Create a *fake* `DefId` out of a `HirId` by combining the owner
        // `local_def_index` and the `local_id`.
        // This will work unless you have *billions* of definitions in a single
        // crate (very unlikely to actually happen).
        rls_data::Id {
            krate: LOCAL_CRATE.as_u32(),
            index: id.owner.local_def_index.as_u32() | id.local_id.as_u32().reverse_bits(),
        }
    })
}

fn null_id() -> rls_data::Id {
    rls_data::Id { krate: u32::MAX, index: u32::MAX }
}

fn lower_attributes(attrs: Vec<ast::Attribute>, scx: &SaveContext<'_>) -> Vec<rls_data::Attribute> {
    attrs
        .into_iter()
        // Only retain real attributes. Doc comments are lowered separately.
        .filter(|attr| !attr.has_name(sym::doc))
        .map(|mut attr| {
            // Remove the surrounding '#[..]' or '#![..]' of the pretty printed
            // attribute. First normalize all inner attribute (#![..]) to outer
            // ones (#[..]), then remove the two leading and the one trailing character.
            attr.style = ast::AttrStyle::Outer;
            let value = attribute_to_string(&attr);
            // This str slicing works correctly, because the leading and trailing characters
            // are in the ASCII range and thus exactly one byte each.
            let value = value[2..value.len() - 1].to_string();

            rls_data::Attribute { value, span: scx.span_from_span(attr.span) }
        })
        .collect()
}
