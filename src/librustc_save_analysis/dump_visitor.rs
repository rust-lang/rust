// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Write the output of rustc's analysis to an implementor of Dump. The data is
//! primarily designed to be used as input to the DXR tool, specifically its
//! Rust plugin. It could also be used by IDEs or other code browsing, search, or
//! cross-referencing tools.
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

use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir::map::{Node, NodeItem};
use rustc::session::Session;
use rustc::ty::{self, TyCtxt, AssociatedItemContainer};

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::*;

use syntax::ast::{self, NodeId, PatKind, Attribute, CRATE_NODE_ID};
use syntax::parse::token;
use syntax::symbol::keywords;
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{path_to_string, ty_to_string, bounds_to_string, generics_to_string};
use syntax::ptr::P;
use syntax::codemap::Spanned;
use syntax_pos::*;

use super::{escape, generated_code, SaveContext, PathCollector, docs_for_attrs};
use super::data::*;
use super::dump::Dump;
use super::external_data::{Lower, make_def_id};
use super::span_utils::SpanUtils;
use super::recorder;

macro_rules! down_cast_data {
    ($id:ident, $kind:ident, $sp:expr) => {
        let $id = if let super::Data::$kind(data) = $id {
            data
        } else {
            span_bug!($sp, "unexpected data kind: {:?}", $id);
        };
    };
}

pub struct DumpVisitor<'l, 'tcx: 'l, 'll, D: 'll> {
    save_ctxt: SaveContext<'l, 'tcx>,
    sess: &'l Session,
    tcx: TyCtxt<'l, 'tcx, 'tcx>,
    dumper: &'ll mut D,

    span: SpanUtils<'l>,

    cur_scope: NodeId,

    // Set of macro definition (callee) spans, and the set
    // of macro use (callsite) spans. We store these to ensure
    // we only write one macro def per unique macro definition, and
    // one macro use per unique callsite span.
    mac_defs: HashSet<Span>,
    mac_uses: HashSet<Span>,
}

impl<'l, 'tcx: 'l, 'll, D: Dump + 'll> DumpVisitor<'l, 'tcx, 'll, D> {
    pub fn new(save_ctxt: SaveContext<'l, 'tcx>,
               dumper: &'ll mut D)
               -> DumpVisitor<'l, 'tcx, 'll, D> {
        let span_utils = SpanUtils::new(&save_ctxt.tcx.sess);
        DumpVisitor {
            sess: &save_ctxt.tcx.sess,
            tcx: save_ctxt.tcx,
            save_ctxt: save_ctxt,
            dumper: dumper,
            span: span_utils.clone(),
            cur_scope: CRATE_NODE_ID,
            mac_defs: HashSet::new(),
            mac_uses: HashSet::new(),
        }
    }

    fn nest_scope<F>(&mut self, scope_id: NodeId, f: F)
        where F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, D>)
    {
        let parent_scope = self.cur_scope;
        self.cur_scope = scope_id;
        f(self);
        self.cur_scope = parent_scope;
    }

    fn nest_tables<F>(&mut self, item_id: NodeId, f: F)
        where F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, D>)
    {
        let old_tables = self.save_ctxt.tables;
        let item_def_id = self.tcx.map.local_def_id(item_id);
        self.save_ctxt.tables = self.tcx.item_tables(item_def_id);
        f(self);
        self.save_ctxt.tables = old_tables;
    }

    pub fn dump_crate_info(&mut self, name: &str, krate: &ast::Crate) {
        let source_file = self.tcx.sess.local_crate_source_file.as_ref();
        let crate_root = source_file.map(|source_file| {
            match source_file.file_name() {
                Some(_) => source_file.parent().unwrap().display().to_string(),
                None => source_file.display().to_string(),
            }
        });

        // Info about all the external crates referenced from this crate.
        let external_crates = self.save_ctxt.get_external_crates().into_iter().map(|c| {
            let lo_loc = self.span.sess.codemap().lookup_char_pos(c.span.lo);
            ExternalCrateData {
                name: c.name,
                num: CrateNum::from_u32(c.number),
                file_name: SpanUtils::make_path_string(&lo_loc.file.name),
            }
        }).collect();

        // The current crate.
        let data = CratePreludeData {
            crate_name: name.into(),
            crate_root: crate_root.unwrap_or("<no source>".to_owned()),
            external_crates: external_crates,
            span: krate.span,
        };

        self.dumper.crate_prelude(data.lower(self.tcx));
    }

    // Return all non-empty prefixes of a path.
    // For each prefix, we return the span for the last segment in the prefix and
    // a str representation of the entire prefix.
    fn process_path_prefixes(&self, path: &ast::Path) -> Vec<(Span, String)> {
        let spans = self.span.spans_for_path_segments(path);
        let segments = &path.segments[if path.is_global() { 1 } else { 0 }..];

        // Paths to enums seem to not match their spans - the span includes all the
        // variants too. But they seem to always be at the end, so I hope we can cope with
        // always using the first ones. So, only error out if we don't have enough spans.
        // What could go wrong...?
        if spans.len() < segments.len() {
            if generated_code(path.span) {
                return vec![];
            }
            error!("Mis-calculated spans for path '{}'. Found {} spans, expected {}. Found spans:",
                   path_to_string(path),
                   spans.len(),
                   segments.len());
            for s in &spans {
                let loc = self.sess.codemap().lookup_char_pos(s.lo);
                error!("    '{}' in {}, line {}",
                       self.span.snippet(*s),
                       loc.file.name,
                       loc.line);
            }
            error!("    master span: {:?}: `{}`", path.span, self.span.snippet(path.span));
            return vec![];
        }

        let mut result: Vec<(Span, String)> = vec![];

        let mut segs = vec![];
        for (i, (seg, span)) in segments.iter().zip(&spans).enumerate() {
            segs.push(seg.clone());
            let sub_path = ast::Path {
                span: *span, // span for the last segment
                segments: segs,
            };
            let qualname = if i == 0 && path.is_global() {
                format!("::{}", path_to_string(&sub_path))
            } else {
                path_to_string(&sub_path)
            };
            result.push((*span, qualname));
            segs = sub_path.segments;
        }

        result
    }

    fn write_sub_paths(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        for (span, qualname) in sub_paths {
            self.dumper.mod_ref(ModRefData {
                span: span,
                qualname: qualname,
                scope: self.cur_scope,
                ref_id: None
            }.lower(self.tcx));
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

        for (span, qualname) in sub_paths.into_iter().take(len - 1) {
            self.dumper.mod_ref(ModRefData {
                span: span,
                qualname: qualname,
                scope: self.cur_scope,
                ref_id: None
            }.lower(self.tcx));
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
        let (ref span, ref qualname) = sub_paths[len-2];
        self.dumper.type_ref(TypeRefData {
            ref_id: None,
            span: *span,
            qualname: qualname.to_owned(),
            scope: CRATE_NODE_ID
        }.lower(self.tcx));

        // write the other sub-paths
        if len <= 2 {
            return;
        }
        let sub_paths = &sub_paths[..len-2];
        for &(ref span, ref qualname) in sub_paths {
            self.dumper.mod_ref(ModRefData {
                span: *span,
                qualname: qualname.to_owned(),
                scope: self.cur_scope,
                ref_id: None
            }.lower(self.tcx));
        }
    }

    fn lookup_def_id(&self, ref_id: NodeId) -> Option<DefId> {
        match self.save_ctxt.get_path_def(ref_id) {
            Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => None,
            def => Some(def.def_id()),
        }
    }

    fn process_def_kind(&mut self,
                        ref_id: NodeId,
                        span: Span,
                        sub_span: Option<Span>,
                        def_id: DefId,
                        scope: NodeId) {
        if self.span.filter_generated(sub_span, span) {
            return;
        }

        let def = self.save_ctxt.get_path_def(ref_id);
        match def {
            Def::Mod(_) => {
                self.dumper.mod_ref(ModRefData {
                    span: sub_span.expect("No span found for mod ref"),
                    ref_id: Some(def_id),
                    scope: scope,
                    qualname: String::new()
                }.lower(self.tcx));
            }
            Def::Struct(..) |
            Def::Variant(..) |
            Def::Union(..) |
            Def::Enum(..) |
            Def::TyAlias(..) |
            Def::Trait(_) => {
                self.dumper.type_ref(TypeRefData {
                    span: sub_span.expect("No span found for type ref"),
                    ref_id: Some(def_id),
                    scope: scope,
                    qualname: String::new()
                }.lower(self.tcx));
            }
            Def::Static(..) |
            Def::Const(..) |
            Def::StructCtor(..) |
            Def::VariantCtor(..) => {
                self.dumper.variable_ref(VariableRefData {
                    span: sub_span.expect("No span found for var ref"),
                    ref_id: def_id,
                    scope: scope,
                    name: String::new()
                }.lower(self.tcx));
            }
            Def::Fn(..) => {
                self.dumper.function_ref(FunctionRefData {
                    span: sub_span.expect("No span found for fn ref"),
                    ref_id: def_id,
                    scope: scope
                }.lower(self.tcx));
            }
            Def::Local(..) |
            Def::Upvar(..) |
            Def::SelfTy(..) |
            Def::Label(_) |
            Def::TyParam(..) |
            Def::Method(..) |
            Def::AssociatedTy(..) |
            Def::AssociatedConst(..) |
            Def::PrimTy(_) |
            Def::Macro(_) |
            Def::Err => {
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
            for &(id, ref p, ..) in &collector.collected_paths {
                let typ = match self.save_ctxt.tables.node_types.get(&id) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                // get the span only for the name of the variable (I hope the path is only ever a
                // variable name, but who knows?)
                let sub_span = span_utils.span_for_last_ident(p.span);
                if !self.span.filter_generated(sub_span, p.span) {
                    self.dumper.variable(VariableData {
                        id: id,
                        kind: VariableKind::Local,
                        span: sub_span.expect("No span found for variable"),
                        name: path_to_string(p),
                        qualname: format!("{}::{}", qualname, path_to_string(p)),
                        type_value: typ,
                        value: String::new(),
                        scope: CRATE_NODE_ID,
                        parent: None,
                        visibility: Visibility::Inherited,
                        docs: String::new(),
                        sig: None,
                    }.lower(self.tcx));
                }
            }
        }
    }

    fn process_method(&mut self,
                      sig: &'l ast::MethodSig,
                      body: Option<&'l ast::Block>,
                      id: ast::NodeId,
                      name: ast::Name,
                      vis: Visibility,
                      attrs: &'l [Attribute],
                      span: Span) {
        debug!("process_method: {}:{}", id, name);

        if let Some(method_data) = self.save_ctxt.get_method_data(id, name, span) {

            let sig_str = ::make_signature(&sig.decl, &sig.generics);
            if body.is_some() {
                self.nest_tables(id, |v| {
                    v.process_formals(&sig.decl.inputs, &method_data.qualname)
                });
            }

            // If the method is defined in an impl, then try and find the corresponding
            // method decl in a trait, and if there is one, make a decl_id for it. This
            // requires looking up the impl, then the trait, then searching for a method
            // with the right name.
            if !self.span.filter_generated(Some(method_data.span), span) {
                let container =
                    self.tcx.associated_item(self.tcx.map.local_def_id(id)).container;
                let mut trait_id;
                let mut decl_id = None;
                match container {
                    AssociatedItemContainer::ImplContainer(id) => {
                        trait_id = self.tcx.trait_id_of_impl(id);

                        match trait_id {
                            Some(id) => {
                                for item in self.tcx.associated_items(id) {
                                    if item.kind == ty::AssociatedKind::Method {
                                        if item.name == name {
                                            decl_id = Some(item.def_id);
                                            break;
                                        }
                                    }
                                }
                            }
                            None => {
                                if let Some(NodeItem(item)) = self.tcx.map.get_if_local(id) {
                                    if let hir::ItemImpl(_, _, _, _, ref ty, _) = item.node {
                                        trait_id = self.lookup_def_id(ty.id);
                                    }
                                }
                            }
                        }
                    }
                    AssociatedItemContainer::TraitContainer(id) => {
                        trait_id = Some(id);
                    }
                }

                self.dumper.method(MethodData {
                    id: method_data.id,
                    name: method_data.name,
                    span: method_data.span,
                    scope: method_data.scope,
                    qualname: method_data.qualname.clone(),
                    value: sig_str,
                    decl_id: decl_id,
                    parent: trait_id,
                    visibility: vis,
                    docs: docs_for_attrs(attrs),
                    sig: method_data.sig,
                }.lower(self.tcx));
            }

            self.process_generic_params(&sig.generics, span, &method_data.qualname, id);
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

    fn process_trait_ref(&mut self, trait_ref: &'l ast::TraitRef) {
        let trait_ref_data = self.save_ctxt.get_trait_ref_data(trait_ref, self.cur_scope);
        if let Some(trait_ref_data) = trait_ref_data {
            if !self.span.filter_generated(Some(trait_ref_data.span), trait_ref.path.span) {
                self.dumper.type_ref(trait_ref_data.lower(self.tcx));
            }
        }
        self.process_path(trait_ref.ref_id, &trait_ref.path, Some(recorder::TypeRef));
    }

    fn process_struct_field_def(&mut self, field: &ast::StructField, parent_id: NodeId) {
        let field_data = self.save_ctxt.get_field_data(field, parent_id);
        if let Some(mut field_data) = field_data {
            if !self.span.filter_generated(Some(field_data.span), field.span) {
                field_data.value = String::new();
                self.dumper.variable(field_data.lower(self.tcx));
            }
        }
    }

    // Dump generic params bindings, then visit_generics
    fn process_generic_params(&mut self,
                              generics: &'l ast::Generics,
                              full_span: Span,
                              prefix: &str,
                              id: NodeId) {
        // We can't only use visit_generics since we don't have spans for param
        // bindings, so we reparse the full_span to get those sub spans.
        // However full span is the entire enum/fn/struct block, so we only want
        // the first few to match the number of generics we're looking for.
        let param_sub_spans = self.span.spans_for_ty_params(full_span,
                                                            (generics.ty_params.len() as isize));
        for (param, param_ss) in generics.ty_params.iter().zip(param_sub_spans) {
            let name = escape(self.span.snippet(param_ss));
            // Append $id to name to make sure each one is unique
            let qualname = format!("{}::{}${}",
                                   prefix,
                                   name,
                                   id);
            if !self.span.filter_generated(Some(param_ss), full_span) {
                self.dumper.typedef(TypeDefData {
                    span: param_ss,
                    name: name,
                    id: param.id,
                    qualname: qualname,
                    value: String::new(),
                    visibility: Visibility::Inherited,
                    parent: None,
                    docs: String::new(),
                    sig: None,
                }.lower(self.tcx));
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
            down_cast_data!(fn_data, FunctionData, item.span);
            if !self.span.filter_generated(Some(fn_data.span), item.span) {
                self.dumper.function(fn_data.clone().lower(self.tcx));
            }

            self.nest_tables(item.id, |v| v.process_formals(&decl.inputs, &fn_data.qualname));
            self.process_generic_params(ty_params, item.span, &fn_data.qualname, item.id);
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
        if let Some(var_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(var_data, VariableData, item.span);
            if !self.span.filter_generated(Some(var_data.span), item.span) {
                self.dumper.variable(var_data.lower(self.tcx));
            }
        }
        self.visit_ty(&typ);
        self.visit_expr(expr);
    }

    fn process_assoc_const(&mut self,
                           id: ast::NodeId,
                           name: ast::Name,
                           span: Span,
                           typ: &'l ast::Ty,
                           expr: &'l ast::Expr,
                           parent_id: DefId,
                           vis: Visibility,
                           attrs: &'l [Attribute]) {
        let qualname = format!("::{}", self.tcx.node_path_str(id));

        let sub_span = self.span.sub_span_after_keyword(span, keywords::Const);

        if !self.span.filter_generated(sub_span, span) {
            self.dumper.variable(VariableData {
                span: sub_span.expect("No span found for variable"),
                kind: VariableKind::Const,
                id: id,
                name: name.to_string(),
                qualname: qualname,
                value: self.span.snippet(expr.span),
                type_value: ty_to_string(&typ),
                scope: self.cur_scope,
                parent: Some(parent_id),
                visibility: vis,
                docs: docs_for_attrs(attrs),
                sig: None,
            }.lower(self.tcx));
        }

        // walk type and init value
        self.visit_ty(typ);
        self.visit_expr(expr);
    }

    // FIXME tuple structs should generate tuple-specific data.
    fn process_struct(&mut self,
                      item: &'l ast::Item,
                      def: &'l ast::VariantData,
                      ty_params: &'l ast::Generics) {
        let name = item.ident.to_string();
        let qualname = format!("::{}", self.tcx.node_path_str(item.id));

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Struct);
        let (val, fields) =
            if let ast::ItemKind::Struct(ast::VariantData::Struct(ref fields, _), _) = item.node
        {
            let fields_str = fields.iter()
                                   .enumerate()
                                   .map(|(i, f)| f.ident.map(|i| i.to_string())
                                                  .unwrap_or(i.to_string()))
                                   .collect::<Vec<_>>()
                                   .join(", ");
            (format!("{} {{ {} }}", name, fields_str), fields.iter().map(|f| f.id).collect())
        } else {
            (String::new(), vec![])
        };

        if !self.span.filter_generated(sub_span, item.span) {
            self.dumper.struct_data(StructData {
                span: sub_span.expect("No span found for struct"),
                id: item.id,
                name: name,
                ctor_id: def.id(),
                qualname: qualname.clone(),
                scope: self.cur_scope,
                value: val,
                fields: fields,
                visibility: From::from(&item.vis),
                docs: docs_for_attrs(&item.attrs),
                sig: self.save_ctxt.sig_base(item),
            }.lower(self.tcx));
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
        down_cast_data!(enum_data, EnumData, item.span);
        if !self.span.filter_generated(Some(enum_data.span), item.span) {
            self.dumper.enum_data(enum_data.clone().lower(self.tcx));
        }

        for variant in &enum_definition.variants {
            let name = variant.node.name.name.to_string();
            let mut qualname = enum_data.qualname.clone();
            qualname.push_str("::");
            qualname.push_str(&name);

            let text = self.span.signature_string_for_span(variant.span);
            let ident_start = text.find(&name).unwrap();
            let ident_end = ident_start + name.len();
            let sig = Signature {
                span: variant.span,
                text: text,
                ident_start: ident_start,
                ident_end: ident_end,
                defs: vec![],
                refs: vec![],
            };

            match variant.node.data {
                ast::VariantData::Struct(ref fields, _) => {
                    let sub_span = self.span.span_for_first_ident(variant.span);
                    let fields_str = fields.iter()
                                           .enumerate()
                                           .map(|(i, f)| f.ident.map(|i| i.to_string())
                                                          .unwrap_or(i.to_string()))
                                           .collect::<Vec<_>>()
                                           .join(", ");
                    let val = format!("{}::{} {{ {} }}", enum_data.name, name, fields_str);
                    if !self.span.filter_generated(sub_span, variant.span) {
                        self.dumper.struct_variant(StructVariantData {
                            span: sub_span.expect("No span found for struct variant"),
                            id: variant.node.data.id(),
                            name: name,
                            qualname: qualname,
                            type_value: enum_data.qualname.clone(),
                            value: val,
                            scope: enum_data.scope,
                            parent: Some(make_def_id(item.id, &self.tcx.map)),
                            docs: docs_for_attrs(&variant.node.attrs),
                            sig: sig,
                        }.lower(self.tcx));
                    }
                }
                ref v => {
                    let sub_span = self.span.span_for_first_ident(variant.span);
                    let mut val = format!("{}::{}", enum_data.name, name);
                    if let &ast::VariantData::Tuple(ref fields, _) = v {
                        val.push('(');
                        val.push_str(&fields.iter()
                                            .map(|f| ty_to_string(&f.ty))
                                            .collect::<Vec<_>>()
                                            .join(", "));
                        val.push(')');
                    }
                    if !self.span.filter_generated(sub_span, variant.span) {
                        self.dumper.tuple_variant(TupleVariantData {
                            span: sub_span.expect("No span found for tuple variant"),
                            id: variant.node.data.id(),
                            name: name,
                            qualname: qualname,
                            type_value: enum_data.qualname.clone(),
                            value: val,
                            scope: enum_data.scope,
                            parent: Some(make_def_id(item.id, &self.tcx.map)),
                            docs: docs_for_attrs(&variant.node.attrs),
                            sig: sig,
                        }.lower(self.tcx));
                    }
                }
            }


            for field in variant.node.data.fields() {
                self.process_struct_field_def(field, variant.node.data.id());
                self.visit_ty(&field.ty);
            }
        }
        self.process_generic_params(ty_params, item.span, &enum_data.qualname, enum_data.id);
    }

    fn process_impl(&mut self,
                    item: &'l ast::Item,
                    type_parameters: &'l ast::Generics,
                    trait_ref: &'l Option<ast::TraitRef>,
                    typ: &'l ast::Ty,
                    impl_items: &'l [ast::ImplItem]) {
        let mut has_self_ref = false;
        if let Some(impl_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(impl_data, ImplData, item.span);
            if let Some(ref self_ref) = impl_data.self_ref {
                has_self_ref = true;
                if !self.span.filter_generated(Some(self_ref.span), item.span) {
                    self.dumper.type_ref(self_ref.clone().lower(self.tcx));
                }
            }
            if let Some(ref trait_ref_data) = impl_data.trait_ref {
                if !self.span.filter_generated(Some(trait_ref_data.span), item.span) {
                    self.dumper.type_ref(trait_ref_data.clone().lower(self.tcx));
                }
            }

            if !self.span.filter_generated(Some(impl_data.span), item.span) {
                self.dumper.impl_data(ImplData {
                    id: impl_data.id,
                    span: impl_data.span,
                    scope: impl_data.scope,
                    trait_ref: impl_data.trait_ref.map(|d| d.ref_id.unwrap()),
                    self_ref: impl_data.self_ref.map(|d| d.ref_id.unwrap())
                }.lower(self.tcx));
            }
        }
        if !has_self_ref {
            self.visit_ty(&typ);
        }
        if let &Some(ref trait_ref) = trait_ref {
            self.process_path(trait_ref.ref_id, &trait_ref.path, Some(recorder::TypeRef));
        }
        self.process_generic_params(type_parameters, item.span, "", item.id);
        for impl_item in impl_items {
            let map = &self.tcx.map;
            self.process_impl_item(impl_item, make_def_id(item.id, map));
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
            self.dumper.trait_data(TraitData {
                span: sub_span.expect("No span found for trait"),
                id: item.id,
                name: name,
                qualname: qualname.clone(),
                scope: self.cur_scope,
                value: val,
                items: methods.iter().map(|i| i.id).collect(),
                visibility: From::from(&item.vis),
                docs: docs_for_attrs(&item.attrs),
                sig: self.save_ctxt.sig_base(item),
            }.lower(self.tcx));
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
                    self.dumper.type_ref(TypeRefData {
                        span: sub_span.expect("No span found for trait ref"),
                        ref_id: Some(id),
                        scope: self.cur_scope,
                        qualname: String::new()
                    }.lower(self.tcx));
                }

                if !self.span.filter_generated(sub_span, trait_ref.path.span) {
                    let sub_span = sub_span.expect("No span for inheritance");
                    self.dumper.inheritance(InheritanceData {
                        span: sub_span,
                        base_id: id,
                        deriv_id: item.id
                    }.lower(self.tcx));
                }
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, item.span, &qualname, item.id);
        for method in methods {
            let map = &self.tcx.map;
            self.process_trait_item(method, make_def_id(item.id, map))
        }
    }

    // `item` is the module in question, represented as an item.
    fn process_mod(&mut self, item: &ast::Item) {
        if let Some(mod_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(mod_data, ModData, item.span);
            if !self.span.filter_generated(Some(mod_data.span), item.span) {
                self.dumper.mod_data(mod_data.lower(self.tcx));
            }
        }
    }

    fn process_path(&mut self, id: NodeId, path: &ast::Path, ref_kind: Option<recorder::Row>) {
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

        match path_data {
            Data::VariableRefData(vrd) => {
                // FIXME: this whole block duplicates the code in process_def_kind
                if !self.span.filter_generated(Some(vrd.span), path.span) {
                    match ref_kind {
                        Some(recorder::TypeRef) => {
                            self.dumper.type_ref(TypeRefData {
                                span: vrd.span,
                                ref_id: Some(vrd.ref_id),
                                scope: vrd.scope,
                                qualname: String::new()
                            }.lower(self.tcx));
                        }
                        Some(recorder::FnRef) => {
                            self.dumper.function_ref(FunctionRefData {
                                span: vrd.span,
                                ref_id: vrd.ref_id,
                                scope: vrd.scope
                            }.lower(self.tcx));
                        }
                        Some(recorder::ModRef) => {
                            self.dumper.mod_ref( ModRefData {
                                span: vrd.span,
                                ref_id: Some(vrd.ref_id),
                                scope: vrd.scope,
                                qualname: String::new()
                            }.lower(self.tcx));
                        }
                        Some(recorder::VarRef) | None
                            => self.dumper.variable_ref(vrd.lower(self.tcx))
                    }
                }

            }
            Data::TypeRefData(trd) => {
                if !self.span.filter_generated(Some(trd.span), path.span) {
                    self.dumper.type_ref(trd.lower(self.tcx));
                }
            }
            Data::MethodCallData(mcd) => {
                if !self.span.filter_generated(Some(mcd.span), path.span) {
                    self.dumper.method_call(mcd.lower(self.tcx));
                }
            }
            Data::FunctionCallData(fcd) => {
                if !self.span.filter_generated(Some(fcd.span), path.span) {
                    self.dumper.function_call(fcd.lower(self.tcx));
                }
            }
            _ => {
               span_bug!(path.span, "Unexpected data: {:?}", path_data);
            }
        }

        // Modules or types in the path prefix.
        match self.save_ctxt.get_path_def(id) {
            Def::Method(did) => {
                let ti = self.tcx.associated_item(did);
                if ti.kind == ty::AssociatedKind::Method && ti.method_has_self_argument {
                    self.write_sub_path_trait_truncated(path);
                }
            }
            Def::Fn(..) |
            Def::Const(..) |
            Def::Static(..) |
            Def::StructCtor(..) |
            Def::VariantCtor(..) |
            Def::AssociatedConst(..) |
            Def::Local(..) |
            Def::Upvar(..) |
            Def::Struct(..) |
            Def::Union(..) |
            Def::Variant(..) |
            Def::TyAlias(..) |
            Def::AssociatedTy(..) => self.write_sub_paths_truncated(path),
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
            down_cast_data!(struct_lit_data, TypeRefData, ex.span);
            if !self.span.filter_generated(Some(struct_lit_data.span), ex.span) {
                self.dumper.type_ref(struct_lit_data.lower(self.tcx));
            }

            let scope = self.save_ctxt.enclosing_scope(ex.id);

            for field in fields {
                if let Some(field_data) = self.save_ctxt
                                              .get_field_ref_data(field, variant, scope) {

                    if !self.span.filter_generated(Some(field_data.span), field.ident.span) {
                        self.dumper.variable_ref(field_data.lower(self.tcx));
                    }
                }

                self.visit_expr(&field.expr)
            }
        }

        walk_list!(self, visit_expr, base);
    }

    fn process_method_call(&mut self, ex: &'l ast::Expr, args: &'l [P<ast::Expr>]) {
        if let Some(mcd) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(mcd, MethodCallData, ex.span);
            if !self.span.filter_generated(Some(mcd.span), ex.span) {
                self.dumper.method_call(mcd.lower(self.tcx));
            }
        }

        // walk receiver and args
        walk_list!(self, visit_expr, args);
    }

    fn process_pat(&mut self, p: &'l ast::Pat) {
        match p.node {
            PatKind::Struct(ref _path, ref fields, _) => {
                // FIXME do something with _path?
                let adt = match self.save_ctxt.tables.node_id_to_type_opt(p.id) {
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
                            self.dumper.variable_ref(VariableRefData {
                                span: sub_span.expect("No span fund for var ref"),
                                ref_id: f.did,
                                scope: self.cur_scope,
                                name: String::new()
                            }.lower(self.tcx));
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

        for &(id, ref p, immut, _) in &collector.collected_paths {
            let mut value = match immut {
                ast::Mutability::Immutable => value.to_string(),
                _ => String::new(),
            };
            let typ = match self.save_ctxt.tables.node_types.get(&id) {
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
            let sub_span = self.span.span_for_last_ident(p.span);
            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            if !self.span.filter_generated(sub_span, p.span) {
                self.dumper.variable(VariableData {
                    span: sub_span.expect("No span found for variable"),
                    kind: VariableKind::Local,
                    id: id,
                    name: path_to_string(p),
                    qualname: format!("{}${}", path_to_string(p), id),
                    value: value,
                    type_value: typ,
                    scope: CRATE_NODE_ID,
                    parent: None,
                    visibility: Visibility::Inherited,
                    docs: String::new(),
                    sig: None,
                }.lower(self.tcx));
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
    fn process_macro_use(&mut self, span: Span, id: NodeId) {
        let data = match self.save_ctxt.get_macro_use_data(span, id) {
            None => return,
            Some(data) => data,
        };
        let mut hasher = DefaultHasher::new();
        data.callee_span.hash(&mut hasher);
        let hash = hasher.finish();
        let qualname = format!("{}::{}", data.name, hash);
        // Don't write macro definition for imported macros
        if !self.mac_defs.contains(&data.callee_span)
            && !data.imported {
            self.mac_defs.insert(data.callee_span);
            if let Some(sub_span) = self.span.span_for_macro_def_name(data.callee_span) {
                self.dumper.macro_data(MacroData {
                    span: sub_span,
                    name: data.name.clone(),
                    qualname: qualname.clone(),
                    // FIXME where do macro docs come from?
                    docs: String::new(),
                }.lower(self.tcx));
            }
        }
        if !self.mac_uses.contains(&data.span) {
            self.mac_uses.insert(data.span);
            if let Some(sub_span) = self.span.span_for_macro_use_name(data.span) {
                self.dumper.macro_use(MacroUseData {
                    span: sub_span,
                    name: data.name,
                    qualname: qualname,
                    scope: data.scope,
                    callee_span: data.callee_span,
                    imported: data.imported,
                }.lower(self.tcx));
            }
        }
    }

    fn process_trait_item(&mut self, trait_item: &'l ast::TraitItem, trait_id: DefId) {
        self.process_macro_use(trait_item.span, trait_item.id);
        match trait_item.node {
            ast::TraitItemKind::Const(ref ty, Some(ref expr)) => {
                self.process_assoc_const(trait_item.id,
                                         trait_item.ident.name,
                                         trait_item.span,
                                         &ty,
                                         &expr,
                                         trait_id,
                                         Visibility::Public,
                                         &trait_item.attrs);
            }
            ast::TraitItemKind::Method(ref sig, ref body) => {
                self.process_method(sig,
                                    body.as_ref().map(|x| &**x),
                                    trait_item.id,
                                    trait_item.ident.name,
                                    Visibility::Public,
                                    &trait_item.attrs,
                                    trait_item.span);
            }
            ast::TraitItemKind::Const(_, None) |
            ast::TraitItemKind::Type(..) |
            ast::TraitItemKind::Macro(_) => {}
        }
    }

    fn process_impl_item(&mut self, impl_item: &'l ast::ImplItem, impl_id: DefId) {
        self.process_macro_use(impl_item.span, impl_item.id);
        match impl_item.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.process_assoc_const(impl_item.id,
                                         impl_item.ident.name,
                                         impl_item.span,
                                         &ty,
                                         &expr,
                                         impl_id,
                                         From::from(&impl_item.vis),
                                         &impl_item.attrs);
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.process_method(sig,
                                    Some(body),
                                    impl_item.id,
                                    impl_item.ident.name,
                                    From::from(&impl_item.vis),
                                    &impl_item.attrs,
                                    impl_item.span);
            }
            ast::ImplItemKind::Type(_) |
            ast::ImplItemKind::Macro(_) => {}
        }
    }
}

impl<'l, 'tcx: 'l, 'll, D: Dump +'ll> Visitor<'l> for DumpVisitor<'l, 'tcx, 'll, D> {
    fn visit_item(&mut self, item: &'l ast::Item) {
        use syntax::ast::ItemKind::*;
        self.process_macro_use(item.span, item.id);
        match item.node {
            Use(ref use_item) => {
                match use_item.node {
                    ast::ViewPathSimple(ident, ref path) => {
                        let sub_span = self.span.span_for_last_ident(path.span);
                        let mod_id = match self.lookup_def_id(item.id) {
                            Some(def_id) => {
                                let scope = self.cur_scope;
                                self.process_def_kind(item.id, path.span, sub_span, def_id, scope);

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
                            self.dumper.use_data(UseData {
                                span: sub_span.expect("No span found for use"),
                                id: item.id,
                                mod_id: mod_id,
                                name: ident.to_string(),
                                scope: self.cur_scope,
                                visibility: From::from(&item.vis),
                            }.lower(self.tcx));
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
                            self.dumper.use_glob(UseGlobData {
                                span: sub_span.expect("No span found for use glob"),
                                id: item.id,
                                names: names,
                                scope: self.cur_scope,
                                visibility: From::from(&item.vis),
                            }.lower(self.tcx));
                        }
                        self.write_sub_paths(path);
                    }
                    ast::ViewPathList(ref path, ref list) => {
                        for plid in list {
                            let scope = self.cur_scope;
                            let id = plid.node.id;
                            if let Some(def_id) = self.lookup_def_id(id) {
                                let span = plid.span;
                                self.process_def_kind(id, span, Some(span), def_id, scope);
                            }
                        }

                        self.write_sub_paths(path);
                    }
                }
            }
            ExternCrate(ref s) => {
                let location = match *s {
                    Some(s) => s.to_string(),
                    None => item.ident.to_string(),
                };
                let alias_span = self.span.span_for_last_ident(item.span);
                let cnum = match self.sess.cstore.extern_mod_stmt_cnum(item.id) {
                    Some(cnum) => cnum,
                    None => LOCAL_CRATE,
                };

                if !self.span.filter_generated(alias_span, item.span) {
                    self.dumper.extern_crate(ExternCrateData {
                        id: item.id,
                        name: item.ident.to_string(),
                        crate_num: cnum,
                        location: location,
                        span: alias_span.expect("No span found for extern crate"),
                        scope: self.cur_scope,
                    }.lower(self.tcx));
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
                    self.dumper.typedef(TypeDefData {
                        span: sub_span.expect("No span found for typedef"),
                        name: item.ident.to_string(),
                        id: item.id,
                        qualname: qualname.clone(),
                        value: value,
                        visibility: From::from(&item.vis),
                        parent: None,
                        docs: docs_for_attrs(&item.attrs),
                        sig: Some(self.save_ctxt.sig_base(item)),
                    }.lower(self.tcx));
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
                    self.process_trait_ref(&trait_ref.trait_ref);
                }
            }
            if let Some(ref ty) = param.default {
                self.visit_ty(&ty);
            }
        }
    }

    fn visit_ty(&mut self, t: &'l ast::Ty) {
        self.process_macro_use(t.span, t.id);
        match t.node {
            ast::TyKind::Path(_, ref path) => {
                if generated_code(t.span) {
                    return;
                }

                if let Some(id) = self.lookup_def_id(t.id) {
                    if let Some(sub_span) = self.span.sub_span_for_type_name(t.span) {
                        self.dumper.type_ref(TypeRefData {
                            span: sub_span,
                            ref_id: Some(id),
                            scope: self.cur_scope,
                            qualname: String::new()
                        }.lower(self.tcx));
                    }
                }

                self.write_sub_paths_truncated(path);
            }
            ast::TyKind::Array(ref element, ref length) => {
                self.visit_ty(element);
                self.nest_tables(length.id, |v| v.visit_expr(length));
            }
            _ => visit::walk_ty(self, t),
        }
    }

    fn visit_expr(&mut self, ex: &'l ast::Expr) {
        self.process_macro_use(ex.span, ex.id);
        match ex.node {
            ast::ExprKind::Call(ref _f, ref _args) => {
                // Don't need to do anything for function calls,
                // because just walking the callee path does what we want.
                visit::walk_expr(self, ex);
            }
            ast::ExprKind::Path(_, ref path) => {
                self.process_path(ex.id, path, None);
                visit::walk_expr(self, ex);
            }
            ast::ExprKind::Struct(ref path, ref fields, ref base) => {
                let hir_expr = self.save_ctxt.tcx.map.expect_expr(ex.id);
                let adt = match self.save_ctxt.tables.expr_ty_opt(&hir_expr) {
                    Some(ty) => ty.ty_adt_def().unwrap(),
                    None => {
                        visit::walk_expr(self, ex);
                        return;
                    }
                };
                let def = self.save_ctxt.get_path_def(hir_expr.id);
                self.process_struct_lit(ex, path, fields, adt.variant_of_def(def), base)
            }
            ast::ExprKind::MethodCall(.., ref args) => self.process_method_call(ex, args),
            ast::ExprKind::Field(ref sub_ex, _) => {
                self.visit_expr(&sub_ex);

                if let Some(field_data) = self.save_ctxt.get_expr_data(ex) {
                    down_cast_data!(field_data, VariableRefData, ex.span);
                    if !self.span.filter_generated(Some(field_data.span), ex.span) {
                        self.dumper.variable_ref(field_data.lower(self.tcx));
                    }
                }
            }
            ast::ExprKind::TupField(ref sub_ex, idx) => {
                self.visit_expr(&sub_ex);

                let hir_node = match self.save_ctxt.tcx.map.find(sub_ex.id) {
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
                            self.dumper.variable_ref(VariableRefData {
                                span: sub_span.expect("No span found for var ref"),
                                ref_id: def.struct_variant().fields[idx.node].did,
                                scope: self.cur_scope,
                                name: String::new()
                            }.lower(self.tcx));
                        }
                    }
                    ty::TyTuple(_) => {}
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
        self.process_macro_use(p.span, p.id);
        self.process_pat(p);
    }

    fn visit_arm(&mut self, arm: &'l ast::Arm) {
        let mut collector = PathCollector::new();
        for pattern in &arm.pats {
            // collect paths from the arm's patterns
            collector.visit_pat(&pattern);
            self.visit_pat(&pattern);
        }

        // This is to get around borrow checking, because we need mut self to call process_path.
        let mut paths_to_process = vec![];

        // process collected paths
        for &(id, ref p, immut, ref_kind) in &collector.collected_paths {
            match self.save_ctxt.get_path_def(id) {
                Def::Local(def_id) => {
                    let id = self.tcx.map.as_local_node_id(def_id).unwrap();
                    let mut value = if immut == ast::Mutability::Immutable {
                        self.span.snippet(p.span).to_string()
                    } else {
                        "<mutable>".to_string()
                    };
                    let typ = self.save_ctxt.tables.node_types
                                  .get(&id).map(|t| t.to_string()).unwrap_or(String::new());
                    value.push_str(": ");
                    value.push_str(&typ);

                    assert!(p.segments.len() == 1,
                            "qualified path for local variable def in arm");
                    if !self.span.filter_generated(Some(p.span), p.span) {
                        self.dumper.variable(VariableData {
                            span: p.span,
                            kind: VariableKind::Local,
                            id: id,
                            name: path_to_string(p),
                            qualname: format!("{}${}", path_to_string(p), id),
                            value: value,
                            type_value: typ,
                            scope: CRATE_NODE_ID,
                            parent: None,
                            visibility: Visibility::Inherited,
                            docs: String::new(),
                            sig: None,
                        }.lower(self.tcx));
                    }
                }
                Def::StructCtor(..) | Def::VariantCtor(..) |
                Def::Const(..) | Def::AssociatedConst(..) |
                Def::Struct(..) | Def::Variant(..) |
                Def::TyAlias(..) | Def::AssociatedTy(..) |
                Def::SelfTy(..) => {
                    paths_to_process.push((id, p.clone(), Some(ref_kind)))
                }
                def => error!("unexpected definition kind when processing collected paths: {:?}",
                              def),
            }
        }

        for &(id, ref path, ref_kind) in &paths_to_process {
            self.process_path(id, path, ref_kind);
        }
        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&arm.body);
    }

    fn visit_stmt(&mut self, s: &'l ast::Stmt) {
        self.process_macro_use(s.span, s.id);
        visit::walk_stmt(self, s)
    }

    fn visit_local(&mut self, l: &'l ast::Local) {
        self.process_macro_use(l.span, l.id);
        let value = l.init.as_ref().map(|i| self.span.snippet(i.span)).unwrap_or(String::new());
        self.process_var_decl(&l.pat, value);

        // Just walk the initialiser and type (don't want to walk the pattern again).
        walk_list!(self, visit_ty, &l.ty);
        walk_list!(self, visit_expr, &l.init);
    }
}
