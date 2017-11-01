// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Write the output of rustc's analysis to an implementor of Dump.
//!
//! Dumping the analysis is implemented by walking the AST and getting a bunch of
//! info out from all over the place. We use Def IDs to identify objects. The
//! tricky part is getting syntactic (span, source text) and semantic (reference
//! Def IDs) information for parts of expressions which the compiler has discarded.
//! E.g., in a path `foo::bar::baz`, the compiler only keeps a span for the whole
//! path and a reference to `baz`, but we want spans and references for all three
//! idents.
//!
//! SpanUtils is used to manipulate spans. In particular, to extract sub-spans
//! from spans (e.g., the span for `bar` from the above example path).
//! DumpVisitor walks the AST and processes it, and an implementor of Dump
//! is used for recording the output in a format-agnostic way (see CsvDumper
//! for an example).

use rustc::hir::def::Def as HirDef;
use rustc::hir::def_id::DefId;
use rustc::hir::map::Node;
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::fx::FxHashSet;

use std::path::Path;

use syntax::ast::{self, NodeId, PatKind, Attribute, CRATE_NODE_ID};
use syntax::parse::token;
use syntax::symbol::keywords;
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{path_to_string, ty_to_string, bounds_to_string, generics_to_string};
use syntax::ptr::P;
use syntax::codemap::Spanned;
use syntax_pos::*;

use {escape, generated_code, SaveContext, PathCollector, lower_attributes};
use json_dumper::{JsonDumper, DumpOutput};
use span_utils::SpanUtils;
use sig;

use rls_data::{CratePreludeData, Import, ImportKind, SpanData, Ref, RefKind,
               Def, DefKind, Relation, RelationKind};

macro_rules! down_cast_data {
    ($id:ident, $kind:ident, $sp:expr) => {
        let $id = if let super::Data::$kind(data) = $id {
            data
        } else {
            span_bug!($sp, "unexpected data kind: {:?}", $id);
        };
    };
}

pub struct DumpVisitor<'l, 'tcx: 'l, 'll, O: DumpOutput + 'll> {
    save_ctxt: SaveContext<'l, 'tcx>,
    tcx: TyCtxt<'l, 'tcx, 'tcx>,
    dumper: &'ll mut JsonDumper<O>,

    span: SpanUtils<'l>,

    cur_scope: NodeId,

    // Set of macro definition (callee) spans, and the set
    // of macro use (callsite) spans. We store these to ensure
    // we only write one macro def per unique macro definition, and
    // one macro use per unique callsite span.
    // mac_defs: HashSet<Span>,
    macro_calls: FxHashSet<Span>,
}

impl<'l, 'tcx: 'l, 'll, O: DumpOutput + 'll> DumpVisitor<'l, 'tcx, 'll, O> {
    pub fn new(save_ctxt: SaveContext<'l, 'tcx>,
               dumper: &'ll mut JsonDumper<O>)
               -> DumpVisitor<'l, 'tcx, 'll, O> {
        let span_utils = SpanUtils::new(&save_ctxt.tcx.sess);
        DumpVisitor {
            tcx: save_ctxt.tcx,
            save_ctxt,
            dumper,
            span: span_utils.clone(),
            cur_scope: CRATE_NODE_ID,
            // mac_defs: HashSet::new(),
            macro_calls: FxHashSet(),
        }
    }

    fn nest_scope<F>(&mut self, scope_id: NodeId, f: F)
        where F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, O>)
    {
        let parent_scope = self.cur_scope;
        self.cur_scope = scope_id;
        f(self);
        self.cur_scope = parent_scope;
    }

    fn nest_tables<F>(&mut self, item_id: NodeId, f: F)
        where F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, O>)
    {
        let item_def_id = self.tcx.hir.local_def_id(item_id);
        if self.tcx.has_typeck_tables(item_def_id) {
            let tables = self.tcx.typeck_tables_of(item_def_id);
            let old_tables = self.save_ctxt.tables;
            self.save_ctxt.tables = tables;
            f(self);
            self.save_ctxt.tables = old_tables;
        } else {
            f(self);
        }
    }

    fn span_from_span(&self, span: Span) -> SpanData {
        self.save_ctxt.span_from_span(span)
    }

    pub fn dump_crate_info(&mut self, name: &str, krate: &ast::Crate) {
        let source_file = self.tcx.sess.local_crate_source_file.as_ref();
        let crate_root = source_file.map(|source_file| {
            let source_file = Path::new(source_file);
            match source_file.file_name() {
                Some(_) => source_file.parent().unwrap().display().to_string(),
                None => source_file.display().to_string(),
            }
        });

        let data = CratePreludeData {
            crate_name: name.into(),
            crate_root: crate_root.unwrap_or("<no source>".to_owned()),
            external_crates: self.save_ctxt.get_external_crates(),
            span: self.span_from_span(krate.span),
        };

        self.dumper.crate_prelude(data);
    }

    // Return all non-empty prefixes of a path.
    // For each prefix, we return the span for the last segment in the prefix and
    // a str representation of the entire prefix.
    fn process_path_prefixes(&self, path: &ast::Path) -> Vec<(Span, String)> {
        let segments = &path.segments[if path.is_global() { 1 } else { 0 }..];

        let mut result = Vec::with_capacity(segments.len());

        let mut segs = vec![];
        for (i, seg) in segments.iter().enumerate() {
            segs.push(seg.clone());
            let sub_path = ast::Path {
                span: seg.span, // span for the last segment
                segments: segs,
            };
            let qualname = if i == 0 && path.is_global() {
                format!("::{}", path_to_string(&sub_path))
            } else {
                path_to_string(&sub_path)
            };
            result.push((seg.span, qualname));
            segs = sub_path.segments;
        }

        result
    }

    fn write_sub_paths(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        for (span, _) in sub_paths {
            let span = self.span_from_span(span);
            self.dumper.dump_ref(Ref {
                kind: RefKind::Mod,
                span,
                ref_id: ::null_id(),
            });
        }
    }

    // As write_sub_paths, but does not process the last ident in the path (assuming it
    // will be processed elsewhere). See note on write_sub_paths about global.
    fn write_sub_paths_truncated(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        let len = sub_paths.len();
        if len <= 1 {
            return;
        }

        for (span, _) in sub_paths.into_iter().take(len - 1) {
            let span = self.span_from_span(span);
            self.dumper.dump_ref(Ref {
                kind: RefKind::Mod,
                span,
                ref_id: ::null_id(),
            });
        }
    }

    // As write_sub_paths, but expects a path of the form module_path::trait::method
    // Where trait could actually be a struct too.
    fn write_sub_path_trait_truncated(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        let len = sub_paths.len();
        if len <= 1 {
            return;
        }
        let sub_paths = &sub_paths[.. (len-1)];

        // write the trait part of the sub-path
        let (ref span, _) = sub_paths[len-2];
        let span = self.span_from_span(*span);
        self.dumper.dump_ref(Ref {
            kind: RefKind::Type,
            ref_id: ::null_id(),
            span,
        });

        // write the other sub-paths
        if len <= 2 {
            return;
        }
        let sub_paths = &sub_paths[..len-2];
        for &(ref span, _) in sub_paths {
            let span = self.span_from_span(*span);
            self.dumper.dump_ref(Ref {
                kind: RefKind::Mod,
                span,
                ref_id: ::null_id(),
            });
        }
    }

    fn lookup_def_id(&self, ref_id: NodeId) -> Option<DefId> {
        match self.save_ctxt.get_path_def(ref_id) {
            HirDef::PrimTy(..) | HirDef::SelfTy(..) | HirDef::Err => None,
            def => Some(def.def_id()),
        }
    }

    fn process_def_kind(&mut self,
                        ref_id: NodeId,
                        span: Span,
                        sub_span: Option<Span>,
                        def_id: DefId) {
        if self.span.filter_generated(sub_span, span) {
            return;
        }

        let def = self.save_ctxt.get_path_def(ref_id);
        match def {
            HirDef::Mod(_) => {
                let span = self.span_from_span(sub_span.expect("No span found for mod ref"));
                self.dumper.dump_ref(Ref {
                    kind: RefKind::Mod,
                    span,
                    ref_id: ::id_from_def_id(def_id),
                });
            }
            HirDef::Struct(..) |
            HirDef::Variant(..) |
            HirDef::Union(..) |
            HirDef::Enum(..) |
            HirDef::TyAlias(..) |
            HirDef::TyForeign(..) |
            HirDef::Trait(_) => {
                let span = self.span_from_span(sub_span.expect("No span found for type ref"));
                self.dumper.dump_ref(Ref {
                    kind: RefKind::Type,
                    span,
                    ref_id: ::id_from_def_id(def_id),
                });
            }
            HirDef::Static(..) |
            HirDef::Const(..) |
            HirDef::StructCtor(..) |
            HirDef::VariantCtor(..) => {
                let span = self.span_from_span(sub_span.expect("No span found for var ref"));
                self.dumper.dump_ref(Ref {
                    kind: RefKind::Variable,
                    span,
                    ref_id: ::id_from_def_id(def_id),
                });
            }
            HirDef::Fn(..) => {
                let span = self.span_from_span(sub_span.expect("No span found for fn ref"));
                self.dumper.dump_ref(Ref {
                    kind: RefKind::Function,
                    span,
                    ref_id: ::id_from_def_id(def_id),
                });
            }
            // With macros 2.0, we can legitimately get a ref to a macro, but
            // we don't handle it properly for now (FIXME).
            HirDef::Macro(..) => {}
            HirDef::Local(..) |
            HirDef::Upvar(..) |
            HirDef::SelfTy(..) |
            HirDef::Label(_) |
            HirDef::TyParam(..) |
            HirDef::Method(..) |
            HirDef::AssociatedTy(..) |
            HirDef::AssociatedConst(..) |
            HirDef::PrimTy(_) |
            HirDef::GlobalAsm(_) |
            HirDef::Err => {
               span_bug!(span,
                         "process_def_kind for unexpected item: {:?}",
                         def);
            }
        }
    }

    fn process_formals(&mut self, formals: &'l [ast::Arg], qualname: &str) {
        for arg in formals {
            self.visit_pat(&arg.pat);
            let mut collector = PathCollector::new();
            collector.visit_pat(&arg.pat);
            let span_utils = self.span.clone();

            for (id, i, sp, ..) in collector.collected_idents {
                let hir_id = self.tcx.hir.node_to_hir_id(id);
                let typ = match self.save_ctxt.tables.node_id_to_type_opt(hir_id) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let sub_span = span_utils.span_for_last_ident(sp);
                if !self.span.filter_generated(sub_span, sp) {
                    let id = ::id_from_node_id(id, &self.save_ctxt);
                    let span = self.span_from_span(sub_span.expect("No span found for variable"));

                    self.dumper.dump_def(false, Def {
                        kind: DefKind::Local,
                        id,
                        span,
                        name: i.to_string(),
                        qualname: format!("{}::{}", qualname, i.to_string()),
                        value: typ,
                        parent: None,
                        children: vec![],
                        decl_id: None,
                        docs: String::new(),
                        sig: None,
                        attributes:vec![],
                    });
                }
            }
        }
    }

    fn process_method(&mut self,
                      sig: &'l ast::MethodSig,
                      body: Option<&'l ast::Block>,
                      id: ast::NodeId,
                      name: ast::Ident,
                      generics: &'l ast::Generics,
                      vis: ast::Visibility,
                      span: Span) {
        debug!("process_method: {}:{}", id, name);

        if let Some(mut method_data) = self.save_ctxt.get_method_data(id, name.name, span) {

            let sig_str = ::make_signature(&sig.decl, &generics);
            if body.is_some() {
                self.nest_tables(id, |v| {
                    v.process_formals(&sig.decl.inputs, &method_data.qualname)
                });
            }

            self.process_generic_params(&generics, span, &method_data.qualname, id);

            method_data.value = sig_str;
            method_data.sig = sig::method_signature(id, name, generics, sig, &self.save_ctxt);
            self.dumper.dump_def(vis == ast::Visibility::Public, method_data);
        }

        // walk arg and return types
        for arg in &sig.decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::FunctionRetTy::Ty(ref ret_ty) = sig.decl.output {
            self.visit_ty(ret_ty);
        }

        // walk the fn body
        if let Some(body) = body {
            self.nest_tables(id, |v| v.nest_scope(id, |v| v.visit_block(body)));
        }
    }

    fn process_struct_field_def(&mut self, field: &ast::StructField, parent_id: NodeId) {
        let field_data = self.save_ctxt.get_field_data(field, parent_id);
        if let Some(field_data) = field_data {
            self.dumper.dump_def(field.vis == ast::Visibility::Public, field_data);
        }
    }

    // Dump generic params bindings, then visit_generics
    fn process_generic_params(&mut self,
                              generics: &'l ast::Generics,
                              full_span: Span,
                              prefix: &str,
                              id: NodeId) {
        for param in &generics.ty_params {
            let param_ss = param.span;
            let name = escape(self.span.snippet(param_ss));
            // Append $id to name to make sure each one is unique
            let qualname = format!("{}::{}${}",
                                   prefix,
                                   name,
                                   id);
            if !self.span.filter_generated(Some(param_ss), full_span) {
                let id = ::id_from_node_id(param.id, &self.save_ctxt);
                let span = self.span_from_span(param_ss);

                self.dumper.dump_def(false, Def {
                    kind: DefKind::Type,
                    id,
                    span,
                    name,
                    qualname,
                    value: String::new(),
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: String::new(),
                    sig: None,
                    attributes: vec![],
                });
            }
        }
        self.visit_generics(generics);
    }

    fn process_fn(&mut self,
                  item: &'l ast::Item,
                  decl: &'l ast::FnDecl,
                  ty_params: &'l ast::Generics,
                  body: &'l ast::Block) {
        if let Some(fn_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(fn_data, DefData, item.span);
            self.nest_tables(item.id, |v| v.process_formals(&decl.inputs, &fn_data.qualname));
            self.process_generic_params(ty_params, item.span, &fn_data.qualname, item.id);
            self.dumper.dump_def(item.vis == ast::Visibility::Public, fn_data);
        }

        for arg in &decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
            self.visit_ty(&ret_ty);
        }

        self.nest_tables(item.id, |v| v.nest_scope(item.id, |v| v.visit_block(&body)));
    }

    fn process_static_or_const_item(&mut self,
                                    item: &'l ast::Item,
                                    typ: &'l ast::Ty,
                                    expr: &'l ast::Expr) {
        self.nest_tables(item.id, |v| {
            if let Some(var_data) = v.save_ctxt.get_item_data(item) {
                down_cast_data!(var_data, DefData, item.span);
                v.dumper.dump_def(item.vis == ast::Visibility::Public, var_data);
            }
            v.visit_ty(&typ);
            v.visit_expr(expr);
        });
    }

    fn process_assoc_const(&mut self,
                           id: ast::NodeId,
                           name: ast::Name,
                           span: Span,
                           typ: &'l ast::Ty,
                           expr: Option<&'l ast::Expr>,
                           parent_id: DefId,
                           vis: ast::Visibility,
                           attrs: &'l [Attribute]) {
        let qualname = format!("::{}", self.tcx.node_path_str(id));

        let sub_span = self.span.sub_span_after_keyword(span, keywords::Const);

        if !self.span.filter_generated(sub_span, span) {
            let sig = sig::assoc_const_signature(id, name, typ, expr, &self.save_ctxt);
            let id = ::id_from_node_id(id, &self.save_ctxt);
            let span = self.span_from_span(sub_span.expect("No span found for variable"));

            self.dumper.dump_def(vis == ast::Visibility::Public, Def {
                kind: DefKind::Const,
                id,
                span,
                name: name.to_string(),
                qualname,
                value: ty_to_string(&typ),
                parent: Some(::id_from_def_id(parent_id)),
                children: vec![],
                decl_id: None,
                docs: self.save_ctxt.docs_for_attrs(attrs),
                sig,
                attributes: lower_attributes(attrs.to_owned(), &self.save_ctxt),
            });
        }

        // walk type and init value
        self.visit_ty(typ);
        if let Some(expr) = expr {
            self.visit_expr(expr);
        }
    }

    // FIXME tuple structs should generate tuple-specific data.
    fn process_struct(&mut self,
                      item: &'l ast::Item,
                      def: &'l ast::VariantData,
                      ty_params: &'l ast::Generics) {
        let name = item.ident.to_string();
        let qualname = format!("::{}", self.tcx.node_path_str(item.id));

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Struct);
        let (value, fields) =
            if let ast::ItemKind::Struct(ast::VariantData::Struct(ref fields, _), _) = item.node
        {
            let include_priv_fields = !self.save_ctxt.config.pub_only;
            let fields_str = fields
                .iter()
                .enumerate()
                .filter_map(|(i, f)| {
                     if include_priv_fields || f.vis == ast::Visibility::Public {
                         f.ident.map(|i| i.to_string()).or_else(|| Some(i.to_string()))
                     } else {
                         None
                     }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let value = format!("{} {{ {} }}", name, fields_str);
            (value, fields.iter().map(|f| ::id_from_node_id(f.id, &self.save_ctxt)).collect())
        } else {
            (String::new(), vec![])
        };

        if !self.span.filter_generated(sub_span, item.span) {
            let span = self.span_from_span(sub_span.expect("No span found for struct"));
            self.dumper.dump_def(item.vis == ast::Visibility::Public, Def {
                kind: DefKind::Struct,
                id: ::id_from_node_id(item.id, &self.save_ctxt),
                span,
                name,
                qualname: qualname.clone(),
                value,
                parent: None,
                children: fields,
                decl_id: None,
                docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                sig: sig::item_signature(item, &self.save_ctxt),
                attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
            });
        }

        for field in def.fields() {
            self.process_struct_field_def(field, item.id);
            self.visit_ty(&field.ty);
        }

        self.process_generic_params(ty_params, item.span, &qualname, item.id);
    }

    fn process_enum(&mut self,
                    item: &'l ast::Item,
                    enum_definition: &'l ast::EnumDef,
                    ty_params: &'l ast::Generics) {
        let enum_data = self.save_ctxt.get_item_data(item);
        let enum_data = match enum_data {
            None => return,
            Some(data) => data,
        };
        down_cast_data!(enum_data, DefData, item.span);

        for variant in &enum_definition.variants {
            let name = variant.node.name.name.to_string();
            let mut qualname = enum_data.qualname.clone();
            qualname.push_str("::");
            qualname.push_str(&name);

            match variant.node.data {
                ast::VariantData::Struct(ref fields, _) => {
                    let sub_span = self.span.span_for_first_ident(variant.span);
                    let fields_str = fields.iter()
                                           .enumerate()
                                           .map(|(i, f)| f.ident.map(|i| i.to_string())
                                                          .unwrap_or(i.to_string()))
                                           .collect::<Vec<_>>()
                                           .join(", ");
                    let value = format!("{}::{} {{ {} }}", enum_data.name, name, fields_str);
                    if !self.span.filter_generated(sub_span, variant.span) {
                        let span = self.span_from_span(
                            sub_span.expect("No span found for struct variant"));
                        let id = ::id_from_node_id(variant.node.data.id(), &self.save_ctxt);
                        let parent = Some(::id_from_node_id(item.id, &self.save_ctxt));

                        self.dumper.dump_def(item.vis == ast::Visibility::Public, Def {
                            kind: DefKind::StructVariant,
                            id,
                            span,
                            name,
                            qualname,
                            value,
                            parent,
                            children: vec![],
                            decl_id: None,
                            docs: self.save_ctxt.docs_for_attrs(&variant.node.attrs),
                            sig: sig::variant_signature(variant, &self.save_ctxt),
                            attributes: lower_attributes(variant.node.attrs.clone(),
                                                         &self.save_ctxt),
                        });
                    }
                }
                ref v => {
                    let sub_span = self.span.span_for_first_ident(variant.span);
                    let mut value = format!("{}::{}", enum_data.name, name);
                    if let &ast::VariantData::Tuple(ref fields, _) = v {
                        value.push('(');
                        value.push_str(&fields.iter()
                                              .map(|f| ty_to_string(&f.ty))
                                              .collect::<Vec<_>>()
                                              .join(", "));
                        value.push(')');
                    }
                    if !self.span.filter_generated(sub_span, variant.span) {
                        let span =
                            self.span_from_span(sub_span.expect("No span found for tuple variant"));
                        let id = ::id_from_node_id(variant.node.data.id(), &self.save_ctxt);
                        let parent = Some(::id_from_node_id(item.id, &self.save_ctxt));

                        self.dumper.dump_def(item.vis == ast::Visibility::Public, Def {
                            kind: DefKind::TupleVariant,
                            id,
                            span,
                            name,
                            qualname,
                            value,
                            parent,
                            children: vec![],
                            decl_id: None,
                            docs: self.save_ctxt.docs_for_attrs(&variant.node.attrs),
                            sig: sig::variant_signature(variant, &self.save_ctxt),
                            attributes: lower_attributes(variant.node.attrs.clone(),
                                                         &self.save_ctxt),
                        });
                    }
                }
            }


            for field in variant.node.data.fields() {
                self.process_struct_field_def(field, variant.node.data.id());
                self.visit_ty(&field.ty);
            }
        }
        self.process_generic_params(ty_params, item.span, &enum_data.qualname, item.id);
        self.dumper.dump_def(item.vis == ast::Visibility::Public, enum_data);
    }

    fn process_impl(&mut self,
                    item: &'l ast::Item,
                    type_parameters: &'l ast::Generics,
                    trait_ref: &'l Option<ast::TraitRef>,
                    typ: &'l ast::Ty,
                    impl_items: &'l [ast::ImplItem]) {
        if let Some(impl_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(impl_data, RelationData, item.span);
            self.dumper.dump_relation(impl_data);
        }
        self.visit_ty(&typ);
        if let &Some(ref trait_ref) = trait_ref {
            self.process_path(trait_ref.ref_id, &trait_ref.path);
        }
        self.process_generic_params(type_parameters, item.span, "", item.id);
        for impl_item in impl_items {
            let map = &self.tcx.hir;
            self.process_impl_item(impl_item, map.local_def_id(item.id));
        }
    }

    fn process_trait(&mut self,
                     item: &'l ast::Item,
                     generics: &'l ast::Generics,
                     trait_refs: &'l ast::TyParamBounds,
                     methods: &'l [ast::TraitItem]) {
        let name = item.ident.to_string();
        let qualname = format!("::{}", self.tcx.node_path_str(item.id));
        let mut val = name.clone();
        if !generics.lifetimes.is_empty() || !generics.ty_params.is_empty() {
            val.push_str(&generics_to_string(generics));
        }
        if !trait_refs.is_empty() {
            val.push_str(": ");
            val.push_str(&bounds_to_string(trait_refs));
        }
        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Trait);
        if !self.span.filter_generated(sub_span, item.span) {
            let id = ::id_from_node_id(item.id, &self.save_ctxt);
            let span = self.span_from_span(sub_span.expect("No span found for trait"));
            let children =
                methods.iter().map(|i| ::id_from_node_id(i.id, &self.save_ctxt)).collect();
            self.dumper.dump_def(item.vis == ast::Visibility::Public, Def {
                kind: DefKind::Trait,
                id,
                span,
                name,
                qualname: qualname.clone(),
                value: val,
                parent: None,
                children,
                decl_id: None,
                docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                sig: sig::item_signature(item, &self.save_ctxt),
                attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
            });
        }

        // super-traits
        for super_bound in trait_refs.iter() {
            let trait_ref = match *super_bound {
                ast::TraitTyParamBound(ref trait_ref, _) => {
                    trait_ref
                }
                ast::RegionTyParamBound(..) => {
                    continue;
                }
            };

            let trait_ref = &trait_ref.trait_ref;
            if let Some(id) = self.lookup_def_id(trait_ref.ref_id) {
                let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                if !self.span.filter_generated(sub_span, trait_ref.path.span) {
                    let span = self.span_from_span(sub_span.expect("No span found for trait ref"));
                    self.dumper.dump_ref(Ref {
                        kind: RefKind::Type,
                        span,
                        ref_id: ::id_from_def_id(id),
                    });
                }

                if !self.span.filter_generated(sub_span, trait_ref.path.span) {
                    let sub_span = self.span_from_span(sub_span.expect("No span for inheritance"));
                    self.dumper.dump_relation(Relation {
                        kind: RelationKind::SuperTrait,
                        span: sub_span,
                        from: ::id_from_def_id(id),
                        to: ::id_from_node_id(item.id, &self.save_ctxt),
                    });
                }
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, item.span, &qualname, item.id);
        for method in methods {
            let map = &self.tcx.hir;
            self.process_trait_item(method, map.local_def_id(item.id))
        }
    }

    // `item` is the module in question, represented as an item.
    fn process_mod(&mut self, item: &ast::Item) {
        if let Some(mod_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(mod_data, DefData, item.span);
            self.dumper.dump_def(item.vis == ast::Visibility::Public, mod_data);
        }
    }

    fn process_path(&mut self, id: NodeId, path: &'l ast::Path) {
        debug!("process_path {:?}", path);
        let path_data = self.save_ctxt.get_path_data(id, path);
        if generated_code(path.span) && path_data.is_none() {
            return;
        }

        let path_data = match path_data {
            Some(pd) => pd,
            None => {
                return;
            }
        };

        self.dumper.dump_ref(path_data);

        // Type parameters
        for seg in &path.segments {
            if let Some(ref params) = seg.parameters {
                match **params {
                    ast::PathParameters::AngleBracketed(ref data) => {
                        for t in &data.types {
                            self.visit_ty(t);
                        }
                    }
                    ast::PathParameters::Parenthesized(ref data) => {
                        for t in &data.inputs {
                            self.visit_ty(t);
                        }
                        if let Some(ref t) = data.output {
                            self.visit_ty(t);
                        }
                    }
                }
            }
        }

        // Modules or types in the path prefix.
        match self.save_ctxt.get_path_def(id) {
            HirDef::Method(did) => {
                let ti = self.tcx.associated_item(did);
                if ti.kind == ty::AssociatedKind::Method && ti.method_has_self_argument {
                    self.write_sub_path_trait_truncated(path);
                }
            }
            HirDef::Fn(..) |
            HirDef::Const(..) |
            HirDef::Static(..) |
            HirDef::StructCtor(..) |
            HirDef::VariantCtor(..) |
            HirDef::AssociatedConst(..) |
            HirDef::Local(..) |
            HirDef::Upvar(..) |
            HirDef::Struct(..) |
            HirDef::Union(..) |
            HirDef::Variant(..) |
            HirDef::TyAlias(..) |
            HirDef::AssociatedTy(..) => self.write_sub_paths_truncated(path),
            _ => {}
        }
    }

    fn process_struct_lit(&mut self,
                          ex: &'l ast::Expr,
                          path: &'l ast::Path,
                          fields: &'l [ast::Field],
                          variant: &'l ty::VariantDef,
                          base: &'l Option<P<ast::Expr>>) {
        self.write_sub_paths_truncated(path);

        if let Some(struct_lit_data) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(struct_lit_data, RefData, ex.span);
            if !generated_code(ex.span) {
                self.dumper.dump_ref(struct_lit_data);
            }

            for field in fields {
                if let Some(field_data) = self.save_ctxt
                                              .get_field_ref_data(field, variant) {
                    self.dumper.dump_ref(field_data);
                }

                self.visit_expr(&field.expr)
            }
        }

        walk_list!(self, visit_expr, base);
    }

    fn process_method_call(&mut self,
                           ex: &'l ast::Expr,
                           seg: &'l ast::PathSegment,
                           args: &'l [P<ast::Expr>]) {
        if let Some(mcd) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(mcd, RefData, ex.span);
            if !generated_code(ex.span) {
                self.dumper.dump_ref(mcd);
            }
        }

        // Explicit types in the turbo-fish.
        if let Some(ref params) = seg.parameters {
            if let ast::PathParameters::AngleBracketed(ref data) = **params {
                for t in &data.types {
                    self.visit_ty(t);
                }
            }
        }

        // walk receiver and args
        walk_list!(self, visit_expr, args);
    }

    fn process_pat(&mut self, p: &'l ast::Pat) {
        match p.node {
            PatKind::Struct(ref _path, ref fields, _) => {
                // FIXME do something with _path?
                let hir_id = self.tcx.hir.node_to_hir_id(p.id);
                let adt = match self.save_ctxt.tables.node_id_to_type_opt(hir_id) {
                    Some(ty) => ty.ty_adt_def().unwrap(),
                    None => {
                        visit::walk_pat(self, p);
                        return;
                    }
                };
                let variant = adt.variant_of_def(self.save_ctxt.get_path_def(p.id));

                for &Spanned { node: ref field, span } in fields {
                    let sub_span = self.span.span_for_first_ident(span);
                    if let Some(f) = variant.find_field_named(field.ident.name) {
                        if !self.span.filter_generated(sub_span, span) {
                            let span =
                                self.span_from_span(sub_span.expect("No span fund for var ref"));
                            self.dumper.dump_ref(Ref {
                                kind: RefKind::Variable,
                                span,
                                ref_id: ::id_from_def_id(f.did),
                            });
                        }
                    }
                    self.visit_pat(&field.pat);
                }
            }
            _ => visit::walk_pat(self, p),
        }
    }


    fn process_var_decl(&mut self, p: &'l ast::Pat, value: String) {
        // The local could declare multiple new vars, we must walk the
        // pattern and collect them all.
        let mut collector = PathCollector::new();
        collector.visit_pat(&p);
        self.visit_pat(&p);

        for (id, i, sp, immut) in collector.collected_idents {
            let mut value = match immut {
                ast::Mutability::Immutable => value.to_string(),
                _ => String::new(),
            };
            let hir_id = self.tcx.hir.node_to_hir_id(id);
            let typ = match self.save_ctxt.tables.node_id_to_type_opt(hir_id) {
                Some(typ) => {
                    let typ = typ.to_string();
                    if !value.is_empty() {
                        value.push_str(": ");
                    }
                    value.push_str(&typ);
                    typ
                }
                None => String::new(),
            };

            // Get the span only for the name of the variable (I hope the path
            // is only ever a variable name, but who knows?).
            let sub_span = self.span.span_for_last_ident(sp);
            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            if !self.span.filter_generated(sub_span, sp) {
                let qualname = format!("{}${}", i.to_string(), id);
                let id = ::id_from_node_id(id, &self.save_ctxt);
                let span = self.span_from_span(sub_span.expect("No span found for variable"));

                self.dumper.dump_def(false, Def {
                    kind: DefKind::Local,
                    id,
                    span,
                    name: i.to_string(),
                    qualname,
                    value: typ,
                    parent: None,
                    children: vec![],
                    decl_id: None,
                    docs: String::new(),
                    sig: None,
                    attributes:vec![],
                });
            }
        }
    }

    /// Extract macro use and definition information from the AST node defined
    /// by the given NodeId, using the expansion information from the node's
    /// span.
    ///
    /// If the span is not macro-generated, do nothing, else use callee and
    /// callsite spans to record macro definition and use data, using the
    /// mac_uses and mac_defs sets to prevent multiples.
    fn process_macro_use(&mut self, span: Span) {
        let source_span = span.source_callsite();
        if self.macro_calls.contains(&source_span) {
            return;
        }
        self.macro_calls.insert(source_span);

        let data = match self.save_ctxt.get_macro_use_data(span) {
            None => return,
            Some(data) => data,
        };

        self.dumper.macro_use(data);

        // FIXME write the macro def
        // let mut hasher = DefaultHasher::new();
        // data.callee_span.hash(&mut hasher);
        // let hash = hasher.finish();
        // let qualname = format!("{}::{}", data.name, hash);
        // Don't write macro definition for imported macros
        // if !self.mac_defs.contains(&data.callee_span)
        //     && !data.imported {
        //     self.mac_defs.insert(data.callee_span);
        //     if let Some(sub_span) = self.span.span_for_macro_def_name(data.callee_span) {
        //         self.dumper.macro_data(MacroData {
        //             span: sub_span,
        //             name: data.name.clone(),
        //             qualname: qualname.clone(),
        //             // FIXME where do macro docs come from?
        //             docs: String::new(),
        //         }.lower(self.tcx));
        //     }
        // }
    }

    fn process_trait_item(&mut self, trait_item: &'l ast::TraitItem, trait_id: DefId) {
        self.process_macro_use(trait_item.span);
        match trait_item.node {
            ast::TraitItemKind::Const(ref ty, ref expr) => {
                self.process_assoc_const(trait_item.id,
                                         trait_item.ident.name,
                                         trait_item.span,
                                         &ty,
                                         expr.as_ref().map(|e| &**e),
                                         trait_id,
                                         ast::Visibility::Public,
                                         &trait_item.attrs);
            }
            ast::TraitItemKind::Method(ref sig, ref body) => {
                self.process_method(sig,
                                    body.as_ref().map(|x| &**x),
                                    trait_item.id,
                                    trait_item.ident,
                                    &trait_item.generics,
                                    ast::Visibility::Public,
                                    trait_item.span);
            }
            ast::TraitItemKind::Type(ref bounds, ref default_ty) => {
                // FIXME do something with _bounds (for type refs)
                let name = trait_item.ident.name.to_string();
                let qualname = format!("::{}", self.tcx.node_path_str(trait_item.id));
                let sub_span = self.span.sub_span_after_keyword(trait_item.span, keywords::Type);

                if !self.span.filter_generated(sub_span, trait_item.span) {
                    let span = self.span_from_span(sub_span.expect("No span found for assoc type"));
                    let id = ::id_from_node_id(trait_item.id, &self.save_ctxt);

                    self.dumper.dump_def(true, Def {
                        kind: DefKind::Type,
                        id,
                        span,
                        name,
                        qualname,
                        value: self.span.snippet(trait_item.span),
                        parent: Some(::id_from_def_id(trait_id)),
                        children: vec![],
                        decl_id: None,
                        docs: self.save_ctxt.docs_for_attrs(&trait_item.attrs),
                        sig: sig::assoc_type_signature(trait_item.id,
                                                       trait_item.ident,
                                                       Some(bounds),
                                                       default_ty.as_ref().map(|ty| &**ty),
                                                       &self.save_ctxt),
                        attributes: lower_attributes(trait_item.attrs.clone(), &self.save_ctxt),
                    });
                }

                if let &Some(ref default_ty) = default_ty {
                    self.visit_ty(default_ty)
                }
            }
            ast::TraitItemKind::Macro(_) => {}
        }
    }

    fn process_impl_item(&mut self, impl_item: &'l ast::ImplItem, impl_id: DefId) {
        self.process_macro_use(impl_item.span);
        match impl_item.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.process_assoc_const(impl_item.id,
                                         impl_item.ident.name,
                                         impl_item.span,
                                         &ty,
                                         Some(expr),
                                         impl_id,
                                         impl_item.vis.clone(),
                                         &impl_item.attrs);
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.process_method(sig,
                                    Some(body),
                                    impl_item.id,
                                    impl_item.ident,
                                    &impl_item.generics,
                                    impl_item.vis.clone(),
                                    impl_item.span);
            }
            ast::ImplItemKind::Type(ref ty) => {
                // FIXME uses of the assoc type should ideally point to this
                // 'def' and the name here should be a ref to the def in the
                // trait.
                self.visit_ty(ty)
            }
            ast::ImplItemKind::Macro(_) => {}
        }
    }
}

impl<'l, 'tcx: 'l, 'll, O: DumpOutput + 'll> Visitor<'l> for DumpVisitor<'l, 'tcx, 'll, O> {
    fn visit_mod(&mut self, m: &'l ast::Mod, span: Span, attrs: &[ast::Attribute], id: NodeId) {
        // Since we handle explicit modules ourselves in visit_item, this should
        // only get called for the root module of a crate.
        assert_eq!(id, ast::CRATE_NODE_ID);

        let qualname = format!("::{}", self.tcx.node_path_str(id));

        let cm = self.tcx.sess.codemap();
        let filename = cm.span_to_filename(span);
        let data_id = ::id_from_node_id(id, &self.save_ctxt);
        let children = m.items.iter().map(|i| ::id_from_node_id(i.id, &self.save_ctxt)).collect();
        let span = self.span_from_span(span);

        self.dumper.dump_def(true, Def {
            kind: DefKind::Mod,
            id: data_id,
            name: String::new(),
            qualname,
            span,
            value: filename,
            children,
            parent: None,
            decl_id: None,
            docs: self.save_ctxt.docs_for_attrs(attrs),
            sig: None,
            attributes: lower_attributes(attrs.to_owned(), &self.save_ctxt),
        });
        self.nest_scope(id, |v| visit::walk_mod(v, m));
    }

    fn visit_item(&mut self, item: &'l ast::Item) {
        use syntax::ast::ItemKind::*;
        self.process_macro_use(item.span);
        match item.node {
            Use(ref use_item) => {
                match use_item.node {
                    ast::ViewPathSimple(ident, ref path) => {
                        let sub_span = self.span.span_for_last_ident(path.span);
                        let mod_id = match self.lookup_def_id(item.id) {
                            Some(def_id) => {
                                self.process_def_kind(item.id, path.span, sub_span, def_id);
                                Some(def_id)
                            }
                            None => None,
                        };

                        // 'use' always introduces an alias, if there is not an explicit
                        // one, there is an implicit one.
                        let sub_span = match self.span.sub_span_after_keyword(use_item.span,
                                                                              keywords::As) {
                            Some(sub_span) => Some(sub_span),
                            None => sub_span,
                        };

                        if !self.span.filter_generated(sub_span, path.span) {
                            let span =
                                self.span_from_span(sub_span.expect("No span found for use"));
                            self.dumper.import(item.vis == ast::Visibility::Public, Import {
                                kind: ImportKind::Use,
                                ref_id: mod_id.map(|id| ::id_from_def_id(id)),
                                span,
                                name: ident.to_string(),
                                value: String::new(),
                            });
                        }
                        self.write_sub_paths_truncated(path);
                    }
                    ast::ViewPathGlob(ref path) => {
                        // Make a comma-separated list of names of imported modules.
                        let mut names = vec![];
                        let glob_map = &self.save_ctxt.analysis.glob_map;
                        let glob_map = glob_map.as_ref().unwrap();
                        if glob_map.contains_key(&item.id) {
                            for n in glob_map.get(&item.id).unwrap() {
                                names.push(n.to_string());
                            }
                        }

                        let sub_span = self.span
                                           .sub_span_of_token(item.span, token::BinOp(token::Star));
                        if !self.span.filter_generated(sub_span, item.span) {
                            let span =
                                self.span_from_span(sub_span.expect("No span found for use glob"));
                            self.dumper.import(item.vis == ast::Visibility::Public, Import {
                                kind: ImportKind::GlobUse,
                                ref_id: None,
                                span,
                                name: "*".to_owned(),
                                value: names.join(", "),
                            });
                        }
                        self.write_sub_paths(path);
                    }
                    ast::ViewPathList(ref path, ref list) => {
                        for plid in list {
                            let id = plid.node.id;
                            if let Some(def_id) = self.lookup_def_id(id) {
                                let span = plid.span;
                                self.process_def_kind(id, span, Some(span), def_id);
                            }
                        }

                        self.write_sub_paths(path);
                    }
                }
            }
            ExternCrate(_) => {
                let alias_span = self.span.span_for_last_ident(item.span);

                if !self.span.filter_generated(alias_span, item.span) {
                    let span =
                        self.span_from_span(alias_span.expect("No span found for extern crate"));
                    self.dumper.import(false, Import {
                        kind: ImportKind::ExternCrate,
                        ref_id: None,
                        span,
                        name: item.ident.to_string(),
                        value: String::new(),
                    });
                }
            }
            Fn(ref decl, .., ref ty_params, ref body) =>
                self.process_fn(item, &decl, ty_params, &body),
            Static(ref typ, _, ref expr) =>
                self.process_static_or_const_item(item, typ, expr),
            Const(ref typ, ref expr) =>
                self.process_static_or_const_item(item, &typ, &expr),
            Struct(ref def, ref ty_params) => self.process_struct(item, def, ty_params),
            Enum(ref def, ref ty_params) => self.process_enum(item, def, ty_params),
            Impl(..,
                 ref ty_params,
                 ref trait_ref,
                 ref typ,
                 ref impl_items) => {
                self.process_impl(item, ty_params, trait_ref, &typ, impl_items)
            }
            Trait(_, ref generics, ref trait_refs, ref methods) =>
                self.process_trait(item, generics, trait_refs, methods),
            Mod(ref m) => {
                self.process_mod(item);
                self.nest_scope(item.id, |v| visit::walk_mod(v, m));
            }
            Ty(ref ty, ref ty_params) => {
                let qualname = format!("::{}", self.tcx.node_path_str(item.id));
                let value = ty_to_string(&ty);
                let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Type);
                if !self.span.filter_generated(sub_span, item.span) {
                    let span = self.span_from_span(sub_span.expect("No span found for typedef"));
                    let id = ::id_from_node_id(item.id, &self.save_ctxt);

                    self.dumper.dump_def(item.vis == ast::Visibility::Public, Def {
                        kind: DefKind::Type,
                        id,
                        span,
                        name: item.ident.to_string(),
                        qualname: qualname.clone(),
                        value,
                        parent: None,
                        children: vec![],
                        decl_id: None,
                        docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                        sig: sig::item_signature(item, &self.save_ctxt),
                        attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
                    });
                }

                self.visit_ty(&ty);
                self.process_generic_params(ty_params, item.span, &qualname, item.id);
            }
            Mac(_) => (),
            _ => visit::walk_item(self, item),
        }
    }

    fn visit_generics(&mut self, generics: &'l ast::Generics) {
        for param in generics.ty_params.iter() {
            for bound in param.bounds.iter() {
                if let ast::TraitTyParamBound(ref trait_ref, _) = *bound {
                    self.process_path(trait_ref.trait_ref.ref_id, &trait_ref.trait_ref.path)
                }
            }
            if let Some(ref ty) = param.default {
                self.visit_ty(&ty);
            }
        }
    }

    fn visit_ty(&mut self, t: &'l ast::Ty) {
        self.process_macro_use(t.span);
        match t.node {
            ast::TyKind::Path(_, ref path) => {
                if generated_code(t.span) {
                    return;
                }

                if let Some(id) = self.lookup_def_id(t.id) {
                    if let Some(sub_span) = self.span.sub_span_for_type_name(t.span) {
                        let span = self.span_from_span(sub_span);
                        self.dumper.dump_ref(Ref {
                            kind: RefKind::Type,
                            span,
                            ref_id: ::id_from_def_id(id),
                        });
                    }
                }

                self.write_sub_paths_truncated(path);
                visit::walk_path(self, path);
            }
            ast::TyKind::Array(ref element, ref length) => {
                self.visit_ty(element);
                self.nest_tables(length.id, |v| v.visit_expr(length));
            }
            _ => visit::walk_ty(self, t),
        }
    }

    fn visit_expr(&mut self, ex: &'l ast::Expr) {
        debug!("visit_expr {:?}", ex.node);
        self.process_macro_use(ex.span);
        match ex.node {
            ast::ExprKind::Struct(ref path, ref fields, ref base) => {
                let hir_expr = self.save_ctxt.tcx.hir.expect_expr(ex.id);
                let adt = match self.save_ctxt.tables.expr_ty_opt(&hir_expr) {
                    Some(ty) if ty.ty_adt_def().is_some() => ty.ty_adt_def().unwrap(),
                    _ => {
                        visit::walk_expr(self, ex);
                        return;
                    }
                };
                let def = self.save_ctxt.get_path_def(hir_expr.id);
                self.process_struct_lit(ex, path, fields, adt.variant_of_def(def), base)
            }
            ast::ExprKind::MethodCall(ref seg, ref args) => self.process_method_call(ex, seg, args),
            ast::ExprKind::Field(ref sub_ex, _) => {
                self.visit_expr(&sub_ex);

                if let Some(field_data) = self.save_ctxt.get_expr_data(ex) {
                    down_cast_data!(field_data, RefData, ex.span);
                    if !generated_code(ex.span) {
                        self.dumper.dump_ref(field_data);
                    }
                }
            }
            ast::ExprKind::TupField(ref sub_ex, idx) => {
                self.visit_expr(&sub_ex);

                let hir_node = match self.save_ctxt.tcx.hir.find(sub_ex.id) {
                    Some(Node::NodeExpr(expr)) => expr,
                    _ => {
                        debug!("Missing or weird node for sub-expression {} in {:?}",
                               sub_ex.id, ex);
                        return;
                    }
                };
                let ty = match self.save_ctxt.tables.expr_ty_adjusted_opt(&hir_node) {
                    Some(ty) => &ty.sty,
                    None => {
                        visit::walk_expr(self, ex);
                        return;
                    }
                };
                match *ty {
                    ty::TyAdt(def, _) => {
                        let sub_span = self.span.sub_span_after_token(ex.span, token::Dot);
                        if !self.span.filter_generated(sub_span, ex.span) {
                            let span =
                                self.span_from_span(sub_span.expect("No span found for var ref"));
                            self.dumper.dump_ref(Ref {
                                kind: RefKind::Variable,
                                span,
                                ref_id: ::id_from_def_id(def.struct_variant().fields[idx.node].did),
                            });
                        }
                    }
                    ty::TyTuple(..) => {}
                    _ => span_bug!(ex.span,
                                   "Expected struct or tuple type, found {:?}",
                                   ty),
                }
            }
            ast::ExprKind::Closure(_, ref decl, ref body, _fn_decl_span) => {
                let mut id = String::from("$");
                id.push_str(&ex.id.to_string());

                // walk arg and return types
                for arg in &decl.inputs {
                    self.visit_ty(&arg.ty);
                }

                if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
                    self.visit_ty(&ret_ty);
                }

                // walk the body
                self.nest_tables(ex.id, |v| {
                    v.process_formals(&decl.inputs, &id);
                    v.nest_scope(ex.id, |v| v.visit_expr(body))
                });
            }
            ast::ExprKind::ForLoop(ref pattern, ref subexpression, ref block, _) |
            ast::ExprKind::WhileLet(ref pattern, ref subexpression, ref block, _) => {
                let value = self.span.snippet(subexpression.span);
                self.process_var_decl(pattern, value);
                debug!("for loop, walk sub-expr: {:?}", subexpression.node);
                visit::walk_expr(self, subexpression);
                visit::walk_block(self, block);
            }
            ast::ExprKind::IfLet(ref pattern, ref subexpression, ref block, ref opt_else) => {
                let value = self.span.snippet(subexpression.span);
                self.process_var_decl(pattern, value);
                visit::walk_expr(self, subexpression);
                visit::walk_block(self, block);
                opt_else.as_ref().map(|el| visit::walk_expr(self, el));
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.visit_expr(element);
                self.nest_tables(count.id, |v| v.visit_expr(count));
            }
            // In particular, we take this branch for call and path expressions,
            // where we'll index the idents involved just by continuing to walk.
            _ => {
                visit::walk_expr(self, ex)
            }
        }
    }

    fn visit_mac(&mut self, mac: &'l ast::Mac) {
        // These shouldn't exist in the AST at this point, log a span bug.
        span_bug!(mac.span, "macro invocation should have been expanded out of AST");
    }

    fn visit_pat(&mut self, p: &'l ast::Pat) {
        self.process_macro_use(p.span);
        self.process_pat(p);
    }

    fn visit_arm(&mut self, arm: &'l ast::Arm) {
        let mut collector = PathCollector::new();
        for pattern in &arm.pats {
            // collect paths from the arm's patterns
            collector.visit_pat(&pattern);
            self.visit_pat(&pattern);
        }

        // process collected paths
        for (id, i, sp, immut) in collector.collected_idents {
            match self.save_ctxt.get_path_def(id) {
                HirDef::Local(id) => {
                    let mut value = if immut == ast::Mutability::Immutable {
                        self.span.snippet(sp).to_string()
                    } else {
                        "<mutable>".to_string()
                    };
                    let hir_id = self.tcx.hir.node_to_hir_id(id);
                    let typ = self.save_ctxt
                                  .tables
                                  .node_id_to_type_opt(hir_id)
                                  .map(|t| t.to_string())
                                  .unwrap_or(String::new());
                    value.push_str(": ");
                    value.push_str(&typ);

                    if !self.span.filter_generated(Some(sp), sp) {
                        let qualname = format!("{}${}", i.to_string(), id);
                        let id = ::id_from_node_id(id, &self.save_ctxt);
                        let span = self.span_from_span(sp);

                        self.dumper.dump_def(false, Def {
                            kind: DefKind::Local,
                            id,
                            span,
                            name: i.to_string(),
                            qualname,
                            value: typ,
                            parent: None,
                            children: vec![],
                            decl_id: None,
                            docs: String::new(),
                            sig: None,
                            attributes:vec![],
                        });
                    }
                }
                def => error!("unexpected definition kind when processing collected idents: {:?}",
                              def),
            }
        }

        for (id, ref path) in collector.collected_paths {
            self.process_path(id, path);
        }
        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&arm.body);
    }

    fn visit_path(&mut self, p: &'l ast::Path, id: NodeId) {
        self.process_path(id, p);
    }

    fn visit_stmt(&mut self, s: &'l ast::Stmt) {
        self.process_macro_use(s.span);
        visit::walk_stmt(self, s)
    }

    fn visit_local(&mut self, l: &'l ast::Local) {
        self.process_macro_use(l.span);
        let value = l.init.as_ref().map(|i| self.span.snippet(i.span)).unwrap_or(String::new());
        self.process_var_decl(&l.pat, value);

        // Just walk the initialiser and type (don't want to walk the pattern again).
        walk_list!(self, visit_ty, &l.ty);
        walk_list!(self, visit_expr, &l.init);
    }

    fn visit_foreign_item(&mut self, item: &'l ast::ForeignItem) {
        match item.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                if let Some(fn_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(fn_data, DefData, item.span);

                    self.nest_tables(item.id, |v| v.process_formals(&decl.inputs,
                                                                    &fn_data.qualname));
                    self.process_generic_params(generics, item.span, &fn_data.qualname, item.id);
                    self.dumper.dump_def(item.vis == ast::Visibility::Public, fn_data);
                }

                for arg in &decl.inputs {
                    self.visit_ty(&arg.ty);
                }

                if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
                    self.visit_ty(&ret_ty);
                }
            }
            ast::ForeignItemKind::Static(ref ty, _) => {
                if let Some(var_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(var_data, DefData, item.span);
                    self.dumper.dump_def(item.vis == ast::Visibility::Public, var_data);
                }

                self.visit_ty(ty);
            }
            ast::ForeignItemKind::Ty => {
                if let Some(var_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(var_data, DefData, item.span);
                    self.dumper.dump_def(item.vis == ast::Visibility::Public, var_data);
                }
            }
        }
    }
}
