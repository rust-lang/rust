// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Output a CSV file containing the output from rustc's analysis. The data is
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
//! Recorder is used for recording the output in csv format. FmtStrs separates
//! the format of the output away from extracting it from the compiler.
//! DumpCsvVisitor walks the AST and processes it.


use super::{escape, generated_code, recorder, SaveContext, PathCollector, Data};

use session::Session;

use middle::def;
use middle::def_id::DefId;
use middle::ty;

use std::fs::File;

use syntax::ast::{self, NodeId};
use syntax::codemap::*;
use syntax::parse::token::{self, keywords};
use syntax::owned_slice::OwnedSlice;
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{path_to_string, ty_to_string};
use syntax::ptr::P;

use rustc_front::lowering::{lower_expr, LoweringContext};

use super::span_utils::SpanUtils;
use super::recorder::{Recorder, FmtStrs};

macro_rules! down_cast_data {
    ($id:ident, $kind:ident, $this:ident, $sp:expr) => {
        let $id = if let super::Data::$kind(data) = $id {
            data
        } else {
            $this.sess.span_bug($sp, &format!("unexpected data kind: {:?}", $id));
        }
    };
}

pub struct DumpCsvVisitor<'l, 'tcx: 'l> {
    save_ctxt: SaveContext<'l, 'tcx>,
    sess: &'l Session,
    tcx: &'l ty::ctxt<'tcx>,
    analysis: &'l ty::CrateAnalysis<'l>,

    span: SpanUtils<'l>,
    fmt: FmtStrs<'l, 'tcx>,

    cur_scope: NodeId,
}

impl <'l, 'tcx> DumpCsvVisitor<'l, 'tcx> {
    pub fn new(tcx: &'l ty::ctxt<'tcx>,
               lcx: &'l LoweringContext<'l>,
               analysis: &'l ty::CrateAnalysis<'l>,
               output_file: Box<File>)
               -> DumpCsvVisitor<'l, 'tcx> {
        let span_utils = SpanUtils::new(&tcx.sess);
        DumpCsvVisitor {
            sess: &tcx.sess,
            tcx: tcx,
            save_ctxt: SaveContext::from_span_utils(tcx, lcx, span_utils.clone()),
            analysis: analysis,
            span: span_utils.clone(),
            fmt: FmtStrs::new(box Recorder {
                                  out: output_file,
                                  dump_spans: false,
                              },
                              span_utils,
                              tcx),
            cur_scope: 0,
        }
    }

    fn nest<F>(&mut self, scope_id: NodeId, f: F)
        where F: FnOnce(&mut DumpCsvVisitor<'l, 'tcx>)
    {
        let parent_scope = self.cur_scope;
        self.cur_scope = scope_id;
        f(self);
        self.cur_scope = parent_scope;
    }

    pub fn dump_crate_info(&mut self, name: &str, krate: &ast::Crate) {
        let source_file = self.tcx.sess.local_crate_source_file.as_ref();
        let crate_root = match source_file {
            Some(source_file) => match source_file.file_name() {
                Some(_) => source_file.parent().unwrap().display().to_string(),
                None => source_file.display().to_string(),
            },
            None => "<no source>".to_owned(),
        };

        // The current crate.
        self.fmt.crate_str(krate.span, name, &crate_root);

        // Dump info about all the external crates referenced from this crate.
        for c in &self.save_ctxt.get_external_crates() {
            self.fmt.external_crate_str(krate.span, &c.name, c.number);
        }
        self.fmt.recorder.record("end_external_crates\n");
    }

    // Return all non-empty prefixes of a path.
    // For each prefix, we return the span for the last segment in the prefix and
    // a str representation of the entire prefix.
    fn process_path_prefixes(&self, path: &ast::Path) -> Vec<(Span, String)> {
        let spans = self.span.spans_for_path_segments(path);

        // Paths to enums seem to not match their spans - the span includes all the
        // variants too. But they seem to always be at the end, so I hope we can cope with
        // always using the first ones. So, only error out if we don't have enough spans.
        // What could go wrong...?
        if spans.len() < path.segments.len() {
            error!("Mis-calculated spans for path '{}'. Found {} spans, expected {}. Found spans:",
                   path_to_string(path),
                   spans.len(),
                   path.segments.len());
            for s in &spans {
                let loc = self.sess.codemap().lookup_char_pos(s.lo);
                error!("    '{}' in {}, line {}",
                       self.span.snippet(*s),
                       loc.file.name,
                       loc.line);
            }
            return vec!();
        }

        let mut result: Vec<(Span, String)> = vec!();

        let mut segs = vec!();
        for (i, (seg, span)) in path.segments.iter().zip(&spans).enumerate() {
            segs.push(seg.clone());
            let sub_path = ast::Path {
                span: *span, // span for the last segment
                global: path.global,
                segments: segs,
            };
            let qualname = if i == 0 && path.global {
                format!("::{}", path_to_string(&sub_path))
            } else {
                path_to_string(&sub_path)
            };
            result.push((*span, qualname));
            segs = sub_path.segments;
        }

        result
    }

    // The global arg allows us to override the global-ness of the path (which
    // actually means 'does the path start with `::`', rather than 'is the path
    // semantically global). We use the override for `use` imports (etc.) where
    // the syntax is non-global, but the semantics are global.
    fn write_sub_paths(&mut self, path: &ast::Path, global: bool) {
        let sub_paths = self.process_path_prefixes(path);
        for (i, &(ref span, ref qualname)) in sub_paths.iter().enumerate() {
            let qualname = if i == 0 && global && !path.global {
                format!("::{}", qualname)
            } else {
                qualname.clone()
            };
            self.fmt.sub_mod_ref_str(path.span, *span, &qualname, self.cur_scope);
        }
    }

    // As write_sub_paths, but does not process the last ident in the path (assuming it
    // will be processed elsewhere). See note on write_sub_paths about global.
    fn write_sub_paths_truncated(&mut self, path: &ast::Path, global: bool) {
        let sub_paths = self.process_path_prefixes(path);
        let len = sub_paths.len();
        if len <= 1 {
            return;
        }

        let sub_paths = &sub_paths[..len-1];
        for (i, &(ref span, ref qualname)) in sub_paths.iter().enumerate() {
            let qualname = if i == 0 && global && !path.global {
                format!("::{}", qualname)
            } else {
                qualname.clone()
            };
            self.fmt.sub_mod_ref_str(path.span, *span, &qualname, self.cur_scope);
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
        self.fmt.sub_type_ref_str(path.span, *span, &qualname);

        // write the other sub-paths
        if len <= 2 {
            return;
        }
        let sub_paths = &sub_paths[..len-2];
        for &(ref span, ref qualname) in sub_paths {
            self.fmt.sub_mod_ref_str(path.span, *span, &qualname, self.cur_scope);
        }
    }

    // looks up anything, not just a type
    fn lookup_type_ref(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.tcx.def_map.borrow().contains_key(&ref_id) {
            self.sess.bug(&format!("def_map has no key for {} in lookup_type_ref",
                                   ref_id));
        }
        let def = self.tcx.def_map.borrow().get(&ref_id).unwrap().full_def();
        match def {
            def::DefPrimTy(..) => None,
            def::DefSelfTy(..) => None,
            _ => Some(def.def_id()),
        }
    }

    fn lookup_def_kind(&self, ref_id: NodeId, span: Span) -> Option<recorder::Row> {
        let def_map = self.tcx.def_map.borrow();
        if !def_map.contains_key(&ref_id) {
            self.sess.span_bug(span,
                               &format!("def_map has no key for {} in lookup_def_kind",
                                        ref_id));
        }
        let def = def_map.get(&ref_id).unwrap().full_def();
        match def {
            def::DefMod(_) |
            def::DefForeignMod(_) => Some(recorder::ModRef),
            def::DefStruct(_) => Some(recorder::TypeRef),
            def::DefTy(..) |
            def::DefAssociatedTy(..) |
            def::DefTrait(_) => Some(recorder::TypeRef),
            def::DefStatic(_, _) |
            def::DefConst(_) |
            def::DefAssociatedConst(..) |
            def::DefLocal(..) |
            def::DefVariant(_, _, _) |
            def::DefUpvar(..) => Some(recorder::VarRef),

            def::DefFn(..) => Some(recorder::FnRef),

            def::DefSelfTy(..) |
            def::DefLabel(_) |
            def::DefTyParam(..) |
            def::DefUse(_) |
            def::DefMethod(..) |
            def::DefPrimTy(_) |
            def::DefErr => {
                self.sess.span_bug(span,
                                   &format!("lookup_def_kind for unexpected item: {:?}", def));
            }
        }
    }

    fn process_formals(&mut self, formals: &Vec<ast::Arg>, qualname: &str) {
        for arg in formals {
            self.visit_pat(&arg.pat);
            let mut collector = PathCollector::new();
            collector.visit_pat(&arg.pat);
            let span_utils = self.span.clone();
            for &(id, ref p, _, _) in &collector.collected_paths {
                let typ = self.tcx.node_types().get(&id).unwrap().to_string();
                // get the span only for the name of the variable (I hope the path is only ever a
                // variable name, but who knows?)
                self.fmt.formal_str(p.span,
                                    span_utils.span_for_last_ident(p.span),
                                    id,
                                    qualname,
                                    &path_to_string(p),
                                    &typ);
            }
        }
    }

    fn process_method(&mut self,
                      sig: &ast::MethodSig,
                      body: Option<&ast::Block>,
                      id: ast::NodeId,
                      name: ast::Name,
                      span: Span) {
        if generated_code(span) {
            return;
        }

        debug!("process_method: {}:{}", id, name);

        let method_data = self.save_ctxt.get_method_data(id, name, span);

        if body.is_some() {
            self.fmt.method_str(span,
                                Some(method_data.span),
                                method_data.id,
                                &method_data.qualname,
                                method_data.declaration,
                                method_data.scope);
            self.process_formals(&sig.decl.inputs, &method_data.qualname);
        } else {
            self.fmt.method_decl_str(span,
                                     Some(method_data.span),
                                     method_data.id,
                                     &method_data.qualname,
                                     method_data.scope);
        }

        // walk arg and return types
        for arg in &sig.decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::Return(ref ret_ty) = sig.decl.output {
            self.visit_ty(ret_ty);
        }

        // walk the fn body
        if let Some(body) = body {
            self.nest(id, |v| v.visit_block(body));
        }

        self.process_generic_params(&sig.generics, span, &method_data.qualname, id);
    }

    fn process_trait_ref(&mut self, trait_ref: &ast::TraitRef) {
        let trait_ref_data = self.save_ctxt.get_trait_ref_data(trait_ref, self.cur_scope);
        if let Some(trait_ref_data) = trait_ref_data {
            self.fmt.ref_str(recorder::TypeRef,
                             trait_ref.path.span,
                             Some(trait_ref_data.span),
                             trait_ref_data.ref_id,
                             trait_ref_data.scope);
            visit::walk_path(self, &trait_ref.path);
        }
    }

    fn process_struct_field_def(&mut self, field: &ast::StructField, parent_id: NodeId) {
        let field_data = self.save_ctxt.get_field_data(field, parent_id);
        if let Some(field_data) = field_data {
            self.fmt.field_str(field.span,
                               Some(field_data.span),
                               field_data.id,
                               &field_data.name,
                               &field_data.qualname,
                               &field_data.type_value,
                               field_data.scope);
        }
    }

    // Dump generic params bindings, then visit_generics
    fn process_generic_params(&mut self,
                              generics: &ast::Generics,
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
            // Append $id to name to make sure each one is unique
            let name = format!("{}::{}${}",
                               prefix,
                               escape(self.span.snippet(param_ss)),
                               id);
            self.fmt.typedef_str(full_span, Some(param_ss), param.id, &name, "");
        }
        self.visit_generics(generics);
    }

    fn process_fn(&mut self,
                  item: &ast::Item,
                  decl: &ast::FnDecl,
                  ty_params: &ast::Generics,
                  body: &ast::Block) {
        let fn_data = self.save_ctxt.get_item_data(item);
        down_cast_data!(fn_data, FunctionData, self, item.span);
        self.fmt.fn_str(item.span,
                        Some(fn_data.span),
                        fn_data.id,
                        &fn_data.qualname,
                        fn_data.scope);


        self.process_formals(&decl.inputs, &fn_data.qualname);
        self.process_generic_params(ty_params, item.span, &fn_data.qualname, item.id);

        for arg in &decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::Return(ref ret_ty) = decl.output {
            self.visit_ty(&ret_ty);
        }

        self.nest(item.id, |v| v.visit_block(&body));
    }

    fn process_static_or_const_item(&mut self, item: &ast::Item, typ: &ast::Ty, expr: &ast::Expr) {
        let var_data = self.save_ctxt.get_item_data(item);
        down_cast_data!(var_data, VariableData, self, item.span);
        self.fmt.static_str(item.span,
                            Some(var_data.span),
                            var_data.id,
                            &var_data.name,
                            &var_data.qualname,
                            &var_data.value,
                            &var_data.type_value,
                            var_data.scope);

        self.visit_ty(&typ);
        self.visit_expr(expr);
    }

    fn process_const(&mut self,
                     id: ast::NodeId,
                     name: ast::Name,
                     span: Span,
                     typ: &ast::Ty,
                     expr: &ast::Expr) {
        let qualname = format!("::{}", self.tcx.map.path_to_string(id));

        let sub_span = self.span.sub_span_after_keyword(span, keywords::Const);

        self.fmt.static_str(span,
                            sub_span,
                            id,
                            &name.as_str(),
                            &qualname,
                            &self.span.snippet(expr.span),
                            &ty_to_string(&*typ),
                            self.cur_scope);

        // walk type and init value
        self.visit_ty(typ);
        self.visit_expr(expr);
    }

    fn process_struct(&mut self,
                      item: &ast::Item,
                      def: &ast::VariantData,
                      ty_params: &ast::Generics) {
        let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));

        let val = self.span.snippet(item.span);
        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Struct);
        self.fmt.struct_str(item.span,
                            sub_span,
                            item.id,
                            def.id(),
                            &qualname,
                            self.cur_scope,
                            &val);

        // fields
        for field in def.fields() {
            self.process_struct_field_def(field, item.id);
            self.visit_ty(&field.node.ty);
        }

        self.process_generic_params(ty_params, item.span, &qualname, item.id);
    }

    fn process_enum(&mut self,
                    item: &ast::Item,
                    enum_definition: &ast::EnumDef,
                    ty_params: &ast::Generics) {
        let enum_data = self.save_ctxt.get_item_data(item);
        down_cast_data!(enum_data, EnumData, self, item.span);
        self.fmt.enum_str(item.span,
                          Some(enum_data.span),
                          enum_data.id,
                          &enum_data.qualname,
                          enum_data.scope,
                          &enum_data.value);

        for variant in &enum_definition.variants {
            let name = &variant.node.name.name.as_str();
            let mut qualname = enum_data.qualname.clone();
            qualname.push_str("::");
            qualname.push_str(name);
            let val = self.span.snippet(variant.span);

            self.fmt.struct_variant_str(variant.span,
                                        self.span.span_for_first_ident(variant.span),
                                        variant.node.data.id(),
                                        variant.node.data.id(),
                                        &qualname,
                                        &enum_data.qualname,
                                        &val,
                                        enum_data.id);

            for field in variant.node.data.fields() {
                self.process_struct_field_def(field, variant.node.data.id());
                self.visit_ty(&*field.node.ty);
            }
        }
        self.process_generic_params(ty_params, item.span, &enum_data.qualname, enum_data.id);
    }

    fn process_impl(&mut self,
                    item: &ast::Item,
                    type_parameters: &ast::Generics,
                    trait_ref: &Option<ast::TraitRef>,
                    typ: &ast::Ty,
                    impl_items: &[P<ast::ImplItem>]) {
        let impl_data = self.save_ctxt.get_item_data(item);
        down_cast_data!(impl_data, ImplData, self, item.span);
        match impl_data.self_ref {
            Some(ref self_ref) => {
                self.fmt.ref_str(recorder::TypeRef,
                                 item.span,
                                 Some(self_ref.span),
                                 self_ref.ref_id,
                                 self_ref.scope);
            }
            None => {
                self.visit_ty(&typ);
            }
        }
        if let Some(ref trait_ref_data) = impl_data.trait_ref {
            self.fmt.ref_str(recorder::TypeRef,
                             item.span,
                             Some(trait_ref_data.span),
                             trait_ref_data.ref_id,
                             trait_ref_data.scope);
            visit::walk_path(self, &trait_ref.as_ref().unwrap().path);
        }

        self.fmt.impl_str(item.span,
                          Some(impl_data.span),
                          impl_data.id,
                          impl_data.self_ref.map(|data| data.ref_id),
                          impl_data.trait_ref.map(|data| data.ref_id),
                          impl_data.scope);

        self.process_generic_params(type_parameters, item.span, "", item.id);
        for impl_item in impl_items {
            self.visit_impl_item(impl_item);
        }
    }

    fn process_trait(&mut self,
                     item: &ast::Item,
                     generics: &ast::Generics,
                     trait_refs: &OwnedSlice<ast::TyParamBound>,
                     methods: &[P<ast::TraitItem>]) {
        let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));
        let val = self.span.snippet(item.span);
        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Trait);
        self.fmt.trait_str(item.span,
                           sub_span,
                           item.id,
                           &qualname,
                           self.cur_scope,
                           &val);

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
            match self.lookup_type_ref(trait_ref.ref_id) {
                Some(id) => {
                    let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                    self.fmt.ref_str(recorder::TypeRef,
                                     trait_ref.path.span,
                                     sub_span,
                                     id,
                                     self.cur_scope);
                    self.fmt.inherit_str(trait_ref.path.span, sub_span, id, item.id);
                }
                None => (),
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, item.span, &qualname, item.id);
        for method in methods {
            self.visit_trait_item(method)
        }
    }

    // `item` is the module in question, represented as an item.
    fn process_mod(&mut self, item: &ast::Item) {
        let mod_data = self.save_ctxt.get_item_data(item);
        down_cast_data!(mod_data, ModData, self, item.span);
        self.fmt.mod_str(item.span,
                         Some(mod_data.span),
                         mod_data.id,
                         &mod_data.qualname,
                         mod_data.scope,
                         &mod_data.filename);
    }

    fn process_path(&mut self, id: NodeId, path: &ast::Path, ref_kind: Option<recorder::Row>) {
        if generated_code(path.span) {
            return;
        }

        let path_data = self.save_ctxt.get_path_data(id, path);
        let path_data = match path_data {
            Some(pd) => pd,
            None => {
                self.tcx.sess.span_bug(path.span,
                                       &format!("Unexpected def kind while looking up path in \
                                                 `{}`",
                                                self.span.snippet(path.span)))
            }
        };
        match path_data {
            Data::VariableRefData(ref vrd) => {
                self.fmt.ref_str(ref_kind.unwrap_or(recorder::VarRef),
                                 path.span,
                                 Some(vrd.span),
                                 vrd.ref_id,
                                 vrd.scope);

            }
            Data::TypeRefData(ref trd) => {
                self.fmt.ref_str(recorder::TypeRef,
                                 path.span,
                                 Some(trd.span),
                                 trd.ref_id,
                                 trd.scope);
            }
            Data::MethodCallData(ref mcd) => {
                self.fmt.meth_call_str(path.span,
                                       Some(mcd.span),
                                       mcd.ref_id,
                                       mcd.decl_id,
                                       mcd.scope);
            }
            Data::FunctionCallData(fcd) => {
                self.fmt.fn_call_str(path.span, Some(fcd.span), fcd.ref_id, fcd.scope);
            }
            _ => {
                self.sess.span_bug(path.span,
                                   &format!("Unexpected data: {:?}", path_data));
            }
        }

        // Modules or types in the path prefix.
        let def_map = self.tcx.def_map.borrow();
        let def = def_map.get(&id).unwrap().full_def();
        match def {
            def::DefMethod(did) => {
                let ti = self.tcx.impl_or_trait_item(did);
                if let ty::MethodTraitItem(m) = ti {
                    if m.explicit_self == ty::StaticExplicitSelfCategory {
                        self.write_sub_path_trait_truncated(path);
                    }
                }
            }
            def::DefLocal(..) |
            def::DefStatic(_,_) |
            def::DefConst(..) |
            def::DefAssociatedConst(..) |
            def::DefStruct(_) |
            def::DefVariant(..) |
            def::DefFn(..) => self.write_sub_paths_truncated(path, false),
            _ => {}
        }
    }

    fn process_struct_lit(&mut self,
                          ex: &ast::Expr,
                          path: &ast::Path,
                          fields: &Vec<ast::Field>,
                          variant: ty::VariantDef,
                          base: &Option<P<ast::Expr>>) {
        if generated_code(path.span) {
            return
        }

        self.write_sub_paths_truncated(path, false);

        if let Some(struct_lit_data) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(struct_lit_data, TypeRefData, self, ex.span);
            self.fmt.ref_str(recorder::TypeRef,
                             ex.span,
                             Some(struct_lit_data.span),
                             struct_lit_data.ref_id,
                             struct_lit_data.scope);
            let scope = self.save_ctxt.enclosing_scope(ex.id);

            for field in fields {
                if generated_code(field.ident.span) {
                    continue;
                }

                let field_data = self.save_ctxt.get_field_ref_data(field, variant, scope);
                self.fmt.ref_str(recorder::VarRef,
                                 field.ident.span,
                                 Some(field_data.span),
                                 field_data.ref_id,
                                 field_data.scope);

                self.visit_expr(&field.expr)
            }
        }

        walk_list!(self, visit_expr, base);
    }

    fn process_method_call(&mut self, ex: &ast::Expr, args: &Vec<P<ast::Expr>>) {
        if let Some(call_data) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(call_data, MethodCallData, self, ex.span);
            self.fmt.meth_call_str(ex.span,
                                   Some(call_data.span),
                                   call_data.ref_id,
                                   call_data.decl_id,
                                   call_data.scope);
        }

        // walk receiver and args
        walk_list!(self, visit_expr, args);
    }

    fn process_pat(&mut self, p: &ast::Pat) {
        if generated_code(p.span) {
            return;
        }

        match p.node {
            ast::PatStruct(ref path, ref fields, _) => {
                visit::walk_path(self, path);
                let adt = self.tcx.node_id_to_type(p.id).ty_adt_def().unwrap();
                let def = self.tcx.def_map.borrow()[&p.id].full_def();
                let variant = adt.variant_of_def(def);

                for &Spanned { node: ref field, span } in fields {
                    if generated_code(span) {
                        continue;
                    }

                    let sub_span = self.span.span_for_first_ident(span);
                    if let Some(f) = variant.find_field_named(field.ident.name) {
                        self.fmt.ref_str(recorder::VarRef, span, sub_span, f.did, self.cur_scope);
                    }
                    self.visit_pat(&field.pat);
                }
            }
            _ => visit::walk_pat(self, p),
        }
    }


    fn process_var_decl(&mut self, p: &ast::Pat, value: String) {
        // The local could declare multiple new vars, we must walk the
        // pattern and collect them all.
        let mut collector = PathCollector::new();
        collector.visit_pat(&p);
        self.visit_pat(&p);

        for &(id, ref p, immut, _) in &collector.collected_paths {
            let value = if immut == ast::MutImmutable {
                value.to_string()
            } else {
                "<mutable>".to_string()
            };
            let types = self.tcx.node_types();
            let typ = types.get(&id).unwrap().to_string();
            // Get the span only for the name of the variable (I hope the path
            // is only ever a variable name, but who knows?).
            let sub_span = self.span.span_for_last_ident(p.span);
            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            self.fmt.variable_str(p.span,
                                  sub_span,
                                  id,
                                  &path_to_string(p),
                                  &value,
                                  &typ);
        }
    }
}

impl<'l, 'tcx, 'v> Visitor<'v> for DumpCsvVisitor<'l, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        if generated_code(item.span) {
            return
        }

        match item.node {
            ast::ItemUse(ref use_item) => {
                match use_item.node {
                    ast::ViewPathSimple(ident, ref path) => {
                        let sub_span = self.span.span_for_last_ident(path.span);
                        let mod_id = match self.lookup_type_ref(item.id) {
                            Some(def_id) => {
                                match self.lookup_def_kind(item.id, path.span) {
                                    Some(kind) => self.fmt.ref_str(kind,
                                                                   path.span,
                                                                   sub_span,
                                                                   def_id,
                                                                   self.cur_scope),
                                    None => {}
                                }
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

                        self.fmt.use_alias_str(path.span,
                                               sub_span,
                                               item.id,
                                               mod_id,
                                               &ident.name.as_str(),
                                               self.cur_scope);
                        self.write_sub_paths_truncated(path, true);
                    }
                    ast::ViewPathGlob(ref path) => {
                        // Make a comma-separated list of names of imported modules.
                        let mut name_string = String::new();
                        let glob_map = &self.analysis.glob_map;
                        let glob_map = glob_map.as_ref().unwrap();
                        if glob_map.contains_key(&item.id) {
                            for n in glob_map.get(&item.id).unwrap() {
                                if !name_string.is_empty() {
                                    name_string.push_str(", ");
                                }
                                name_string.push_str(&n.as_str());
                            }
                        }

                        let sub_span = self.span
                                           .sub_span_of_token(path.span, token::BinOp(token::Star));
                        self.fmt.use_glob_str(path.span,
                                              sub_span,
                                              item.id,
                                              &name_string,
                                              self.cur_scope);
                        self.write_sub_paths(path, true);
                    }
                    ast::ViewPathList(ref path, ref list) => {
                        for plid in list {
                            match plid.node {
                                ast::PathListIdent { id, .. } => {
                                    match self.lookup_type_ref(id) {
                                        Some(def_id) => match self.lookup_def_kind(id, plid.span) {
                                            Some(kind) => {
                                                self.fmt.ref_str(kind,
                                                                 plid.span,
                                                                 Some(plid.span),
                                                                 def_id,
                                                                 self.cur_scope);
                                            }
                                            None => (),
                                        },
                                        None => (),
                                    }
                                }
                                ast::PathListMod { .. } => (),
                            }
                        }

                        self.write_sub_paths(path, true);
                    }
                }
            }
            ast::ItemExternCrate(ref s) => {
                let location = match *s {
                    Some(s) => s.to_string(),
                    None => item.ident.to_string(),
                };
                let alias_span = self.span.span_for_last_ident(item.span);
                let cnum = match self.sess.cstore.extern_mod_stmt_cnum(item.id) {
                    Some(cnum) => cnum,
                    None => 0,
                };
                self.fmt.extern_crate_str(item.span,
                                          alias_span,
                                          item.id,
                                          cnum,
                                          &item.ident.name.as_str(),
                                          &location,
                                          self.cur_scope);
            }
            ast::ItemFn(ref decl, _, _, _, ref ty_params, ref body) =>
                self.process_fn(item, &**decl, ty_params, &**body),
            ast::ItemStatic(ref typ, _, ref expr) =>
                self.process_static_or_const_item(item, typ, expr),
            ast::ItemConst(ref typ, ref expr) =>
                self.process_static_or_const_item(item, &typ, &expr),
            ast::ItemStruct(ref def, ref ty_params) => self.process_struct(item, def, ty_params),
            ast::ItemEnum(ref def, ref ty_params) => self.process_enum(item, def, ty_params),
            ast::ItemImpl(_, _,
                          ref ty_params,
                          ref trait_ref,
                          ref typ,
                          ref impl_items) => {
                self.process_impl(item, ty_params, trait_ref, &typ, impl_items)
            }
            ast::ItemTrait(_, ref generics, ref trait_refs, ref methods) =>
                self.process_trait(item, generics, trait_refs, methods),
            ast::ItemMod(ref m) => {
                self.process_mod(item);
                self.nest(item.id, |v| visit::walk_mod(v, m));
            }
            ast::ItemTy(ref ty, ref ty_params) => {
                let qualname = format!("::{}", self.tcx.map.path_to_string(item.id));
                let value = ty_to_string(&**ty);
                let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Type);
                self.fmt.typedef_str(item.span, sub_span, item.id, &qualname, &value);

                self.visit_ty(&**ty);
                self.process_generic_params(ty_params, item.span, &qualname, item.id);
            }
            ast::ItemMac(_) => (),
            _ => visit::walk_item(self, item),
        }
    }

    fn visit_generics(&mut self, generics: &ast::Generics) {
        for param in generics.ty_params.iter() {
            for bound in param.bounds.iter() {
                if let ast::TraitTyParamBound(ref trait_ref, _) = *bound {
                    self.process_trait_ref(&trait_ref.trait_ref);
                }
            }
            if let Some(ref ty) = param.default {
                self.visit_ty(&**ty);
            }
        }
    }

    fn visit_trait_item(&mut self, trait_item: &ast::TraitItem) {
        match trait_item.node {
            ast::ConstTraitItem(ref ty, Some(ref expr)) => {
                self.process_const(trait_item.id,
                                   trait_item.ident.name,
                                   trait_item.span,
                                   &*ty,
                                   &*expr);
            }
            ast::MethodTraitItem(ref sig, ref body) => {
                self.process_method(sig,
                                    body.as_ref().map(|x| &**x),
                                    trait_item.id,
                                    trait_item.ident.name,
                                    trait_item.span);
            }
            ast::ConstTraitItem(_, None) |
            ast::TypeTraitItem(..) => {}
        }
    }

    fn visit_impl_item(&mut self, impl_item: &ast::ImplItem) {
        match impl_item.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.process_const(impl_item.id,
                                   impl_item.ident.name,
                                   impl_item.span,
                                   &ty,
                                   &expr);
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.process_method(sig,
                                    Some(body),
                                    impl_item.id,
                                    impl_item.ident.name,
                                    impl_item.span);
            }
            ast::ImplItemKind::Type(_) |
            ast::ImplItemKind::Macro(_) => {}
        }
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        if generated_code(t.span) {
            return
        }

        match t.node {
            ast::TyPath(_, ref path) => {
                match self.lookup_type_ref(t.id) {
                    Some(id) => {
                        let sub_span = self.span.sub_span_for_type_name(t.span);
                        self.fmt.ref_str(recorder::TypeRef, t.span, sub_span, id, self.cur_scope);
                    }
                    None => (),
                }

                self.write_sub_paths_truncated(path, false);

                visit::walk_path(self, path);
            }
            _ => visit::walk_ty(self, t),
        }
    }

    fn visit_expr(&mut self, ex: &ast::Expr) {
        if generated_code(ex.span) {
            return
        }

        match ex.node {
            ast::ExprCall(ref _f, ref _args) => {
                // Don't need to do anything for function calls,
                // because just walking the callee path does what we want.
                visit::walk_expr(self, ex);
            }
            ast::ExprPath(_, ref path) => {
                self.process_path(ex.id, path, None);
                visit::walk_expr(self, ex);
            }
            ast::ExprStruct(ref path, ref fields, ref base) => {
                let hir_expr = lower_expr(self.save_ctxt.lcx, ex);
                let adt = self.tcx.expr_ty(&hir_expr).ty_adt_def().unwrap();
                let def = self.tcx.resolve_expr(&hir_expr);
                self.process_struct_lit(ex, path, fields, adt.variant_of_def(def), base)
            }
            ast::ExprMethodCall(_, _, ref args) => self.process_method_call(ex, args),
            ast::ExprField(ref sub_ex, _) => {
                if generated_code(sub_ex.span) {
                    return
                }

                self.visit_expr(&sub_ex);

                if let Some(field_data) = self.save_ctxt.get_expr_data(ex) {
                    down_cast_data!(field_data, VariableRefData, self, ex.span);
                    self.fmt.ref_str(recorder::VarRef,
                                     ex.span,
                                     Some(field_data.span),
                                     field_data.ref_id,
                                     field_data.scope);
                }
            }
            ast::ExprTupField(ref sub_ex, idx) => {
                if generated_code(sub_ex.span) {
                    return
                }

                self.visit_expr(&**sub_ex);

                let hir_node = lower_expr(self.save_ctxt.lcx, sub_ex);
                let ty = &self.tcx.expr_ty_adjusted(&hir_node).sty;
                match *ty {
                    ty::TyStruct(def, _) => {
                        let sub_span = self.span.sub_span_after_token(ex.span, token::Dot);
                        self.fmt.ref_str(recorder::VarRef,
                                         ex.span,
                                         sub_span,
                                         def.struct_variant().fields[idx.node].did,
                                         self.cur_scope);
                    }
                    ty::TyTuple(_) => {}
                    _ => self.sess.span_bug(ex.span,
                                            &format!("Expected struct or tuple type, found {:?}",
                                                     ty)),
                }
            }
            ast::ExprClosure(_, ref decl, ref body) => {
                if generated_code(body.span) {
                    return
                }

                let mut id = String::from("$");
                id.push_str(&ex.id.to_string());
                self.process_formals(&decl.inputs, &id);

                // walk arg and return types
                for arg in &decl.inputs {
                    self.visit_ty(&*arg.ty);
                }

                if let ast::Return(ref ret_ty) = decl.output {
                    self.visit_ty(&**ret_ty);
                }

                // walk the body
                self.nest(ex.id, |v| v.visit_block(&**body));
            }
            ast::ExprForLoop(ref pattern, ref subexpression, ref block, _) |
            ast::ExprWhileLet(ref pattern, ref subexpression, ref block, _) => {
                let value = self.span.snippet(mk_sp(ex.span.lo, subexpression.span.hi));
                self.process_var_decl(pattern, value);
                visit::walk_expr(self, subexpression);
                visit::walk_block(self, block);
            }
            ast::ExprIfLet(ref pattern, ref subexpression, ref block, ref opt_else) => {
                let value = self.span.snippet(mk_sp(ex.span.lo, subexpression.span.hi));
                self.process_var_decl(pattern, value);
                visit::walk_expr(self, subexpression);
                visit::walk_block(self, block);
                opt_else.as_ref().map(|el| visit::walk_expr(self, el));
            }
            _ => {
                visit::walk_expr(self, ex)
            }
        }
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // Just stop, macros are poison to us.
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        self.process_pat(p);
    }

    fn visit_arm(&mut self, arm: &ast::Arm) {
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
            let def_map = self.tcx.def_map.borrow();
            if !def_map.contains_key(&id) {
                self.sess.span_bug(p.span,
                                   &format!("def_map has no key for {} in visit_arm", id));
            }
            let def = def_map.get(&id).unwrap().full_def();
            match def {
                def::DefLocal(_, id) => {
                    let value = if immut == ast::MutImmutable {
                        self.span.snippet(p.span).to_string()
                    } else {
                        "<mutable>".to_string()
                    };

                    assert!(p.segments.len() == 1,
                            "qualified path for local variable def in arm");
                    self.fmt.variable_str(p.span, Some(p.span), id, &path_to_string(p), &value, "")
                }
                def::DefVariant(..) | def::DefTy(..) | def::DefStruct(..) => {
                    paths_to_process.push((id, p.clone(), Some(ref_kind)))
                }
                // FIXME(nrc) what are these doing here?
                def::DefStatic(_, _) |
                def::DefConst(..) |
                def::DefAssociatedConst(..) => {}
                _ => error!("unexpected definition kind when processing collected paths: {:?}",
                            def),
            }
        }

        for &(id, ref path, ref_kind) in &paths_to_process {
            self.process_path(id, path, ref_kind);
        }
        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&arm.body);
    }

    fn visit_stmt(&mut self, s: &ast::Stmt) {
        if generated_code(s.span) {
            return
        }

        visit::walk_stmt(self, s)
    }

    fn visit_local(&mut self, l: &ast::Local) {
        if generated_code(l.span) {
            return
        }

        let value = self.span.snippet(l.span);
        self.process_var_decl(&l.pat, value);

        // Just walk the initialiser and type (don't want to walk the pattern again).
        walk_list!(self, visit_ty, &l.ty);
        walk_list!(self, visit_expr, &l.init);
    }
}
