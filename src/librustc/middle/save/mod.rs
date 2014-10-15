// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
//! DxrVisitor walks the AST and processes it.

use driver::driver::CrateAnalysis;
use driver::session::Session;

use middle::def;
use middle::ty;
use middle::typeck;

use std::cell::Cell;
use std::io;
use std::io::File;
use std::io::fs;
use std::os;

use syntax::ast;
use syntax::ast_util;
use syntax::ast_util::PostExpansionMethod;
use syntax::ast::{NodeId,DefId};
use syntax::ast_map::NodeItem;
use syntax::attr;
use syntax::codemap::*;
use syntax::parse::token;
use syntax::parse::token::{get_ident,keywords};
use syntax::owned_slice::OwnedSlice;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::print::pprust::{path_to_string,ty_to_string};
use syntax::ptr::P;

use middle::save::span_utils::SpanUtils;
use middle::save::recorder::Recorder;
use middle::save::recorder::FmtStrs;

use util::ppaux;

mod span_utils;
mod recorder;

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

// If the expression is a macro expansion or other generated code, run screaming and don't index.
fn generated_code(span: Span) -> bool {
    span.expn_id != NO_EXPANSION || span  == DUMMY_SP
}

struct DxrVisitor<'l, 'tcx: 'l> {
    sess: &'l Session,
    analysis: &'l CrateAnalysis<'tcx>,

    collected_paths: Vec<(NodeId, ast::Path, bool, recorder::Row)>,
    collecting: bool,

    span: SpanUtils<'l>,
    fmt: FmtStrs<'l>,

    cur_scope: NodeId
}

impl <'l, 'tcx> DxrVisitor<'l, 'tcx> {
    fn nest(&mut self, scope_id: NodeId, f: |&mut DxrVisitor<'l, 'tcx>|) {
        let parent_scope = self.cur_scope;
        self.cur_scope = scope_id;
        f(self);
        self.cur_scope = parent_scope;
    }

    fn dump_crate_info(&mut self, name: &str, krate: &ast::Crate) {
        // the current crate
        self.fmt.crate_str(krate.span, name);

        // dump info about all the external crates referenced from this crate
        self.sess.cstore.iter_crate_data(|n, cmd| {
            self.fmt.external_crate_str(krate.span, cmd.name.as_slice(), n);
        });
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
            error!("Mis-calculated spans for path '{}'. \
                    Found {} spans, expected {}. Found spans:",
                   path_to_string(path), spans.len(), path.segments.len());
            for s in spans.iter() {
                let loc = self.sess.codemap().lookup_char_pos(s.lo);
                error!("    '{}' in {}, line {}",
                       self.span.snippet(*s), loc.file.name, loc.line);
            }
            return vec!();
        }

        let mut result: Vec<(Span, String)> = vec!();


        let mut segs = vec!();
        for (seg, span) in path.segments.iter().zip(spans.iter()) {
            segs.push(seg.clone());
            let sub_path = ast::Path{span: *span, // span for the last segment
                                     global: path.global,
                                     segments: segs};
            let qualname = path_to_string(&sub_path);
            result.push((*span, qualname));
            segs = sub_path.segments;
        }

        result
    }

    fn write_sub_paths(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        for &(ref span, ref qualname) in sub_paths.iter() {
            self.fmt.sub_mod_ref_str(path.span,
                                     *span,
                                     qualname.as_slice(),
                                     self.cur_scope);
        }
    }

    // As write_sub_paths, but does not process the last ident in the path (assuming it
    // will be processed elsewhere).
    fn write_sub_paths_truncated(&mut self, path: &ast::Path) {
        let sub_paths = self.process_path_prefixes(path);
        let len = sub_paths.len();
        if len <= 1 {
            return;
        }

        let sub_paths = sub_paths.slice(0, len-1);
        for &(ref span, ref qualname) in sub_paths.iter() {
            self.fmt.sub_mod_ref_str(path.span,
                                     *span,
                                     qualname.as_slice(),
                                     self.cur_scope);
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
        let sub_paths = sub_paths.slice_to(len-1);

        // write the trait part of the sub-path
        let (ref span, ref qualname) = sub_paths[len-2];
        self.fmt.sub_type_ref_str(path.span,
                                  *span,
                                  qualname.as_slice());

        // write the other sub-paths
        if len <= 2 {
            return;
        }
        let sub_paths = sub_paths[..len-2];
        for &(ref span, ref qualname) in sub_paths.iter() {
            self.fmt.sub_mod_ref_str(path.span,
                                     *span,
                                     qualname.as_slice(),
                                     self.cur_scope);
        }
    }

    // looks up anything, not just a type
    fn lookup_type_ref(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.analysis.ty_cx.def_map.borrow().contains_key(&ref_id) {
            self.sess.bug(format!("def_map has no key for {} in lookup_type_ref",
                                  ref_id).as_slice());
        }
        let def = (*self.analysis.ty_cx.def_map.borrow())[ref_id];
        match def {
            def::DefPrimTy(_) => None,
            _ => Some(def.def_id()),
        }
    }

    fn lookup_def_kind(&self, ref_id: NodeId, span: Span) -> Option<recorder::Row> {
        let def_map = self.analysis.ty_cx.def_map.borrow();
        if !def_map.contains_key(&ref_id) {
            self.sess.span_bug(span, format!("def_map has no key for {} in lookup_def_kind",
                                             ref_id).as_slice());
        }
        let def = (*def_map)[ref_id];
        match def {
            def::DefMod(_) |
            def::DefForeignMod(_) => Some(recorder::ModRef),
            def::DefStruct(_) => Some(recorder::StructRef),
            def::DefTy(..) |
            def::DefAssociatedTy(..) |
            def::DefTrait(_) => Some(recorder::TypeRef),
            def::DefStatic(_, _) |
            def::DefConst(_) |
            def::DefLocal(_) |
            def::DefVariant(_, _, _) |
            def::DefUpvar(..) => Some(recorder::VarRef),

            def::DefFn(..) => Some(recorder::FnRef),

            def::DefSelfTy(_) |
            def::DefRegion(_) |
            def::DefTyParamBinder(_) |
            def::DefLabel(_) |
            def::DefStaticMethod(_, _, _) |
            def::DefTyParam(..) |
            def::DefUse(_) |
            def::DefMethod(..) |
            def::DefPrimTy(_) => {
                self.sess.span_bug(span, format!("lookup_def_kind for unexpected item: {}",
                                                 def).as_slice());
            },
        }
    }

    fn process_formals(&mut self, formals: &Vec<ast::Arg>, qualname: &str) {
        for arg in formals.iter() {
            assert!(self.collected_paths.len() == 0 && !self.collecting);
            self.collecting = true;
            self.visit_pat(&*arg.pat);
            self.collecting = false;
            let span_utils = self.span;
            for &(id, ref p, _, _) in self.collected_paths.iter() {
                let typ = ppaux::ty_to_string(&self.analysis.ty_cx,
                    (*self.analysis.ty_cx.node_types.borrow())[id as uint]);
                // get the span only for the name of the variable (I hope the path is only ever a
                // variable name, but who knows?)
                self.fmt.formal_str(p.span,
                                    span_utils.span_for_last_ident(p.span),
                                    id,
                                    qualname,
                                    path_to_string(p).as_slice(),
                                    typ.as_slice());
            }
            self.collected_paths.clear();
        }
    }

    fn process_method(&mut self, method: &ast::Method) {
        if generated_code(method.span) {
            return;
        }

        let mut scope_id;
        // The qualname for a method is the trait name or name of the struct in an impl in
        // which the method is declared in followed by the method's name.
        let mut qualname = match ty::impl_of_method(&self.analysis.ty_cx,
                                                ast_util::local_def(method.id)) {
            Some(impl_id) => match self.analysis.ty_cx.map.get(impl_id.node) {
                NodeItem(item) => {
                    scope_id = item.id;
                    match item.node {
                        ast::ItemImpl(_, _, ref ty, _) => {
                            let mut result = String::from_str("<");
                            result.push_str(ty_to_string(&**ty).as_slice());

                            match ty::trait_of_item(&self.analysis.ty_cx,
                                                    ast_util::local_def(method.id)) {
                                Some(def_id) => {
                                    result.push_str(" as ");
                                    result.push_str(
                                        ty::item_path_str(&self.analysis.ty_cx, def_id).as_slice());
                                },
                                None => {}
                            }
                            result.push_str(">::");
                            result
                        }
                        _ => {
                            self.sess.span_bug(method.span,
                                               format!("Container {} for method {} not an impl?",
                                                       impl_id.node, method.id).as_slice());
                        },
                    }
                },
                _ => {
                    self.sess.span_bug(method.span,
                                       format!("Container {} for method {} is not a node item {}",
                                               impl_id.node,
                                               method.id,
                                               self.analysis.ty_cx.map.get(impl_id.node)
                                              ).as_slice());
                },
            },
            None => match ty::trait_of_item(&self.analysis.ty_cx,
                                            ast_util::local_def(method.id)) {
                Some(def_id) => {
                    scope_id = def_id.node;
                    match self.analysis.ty_cx.map.get(def_id.node) {
                        NodeItem(_) => {
                            let mut result = ty::item_path_str(&self.analysis.ty_cx, def_id);
                            result.push_str("::");
                            result
                        }
                        _ => {
                            self.sess.span_bug(method.span,
                                               format!("Could not find container {} for method {}",
                                                       def_id.node, method.id).as_slice());
                        }
                    }
                },
                None => {
                    self.sess.span_bug(method.span,
                                       format!("Could not find container for method {}",
                                               method.id).as_slice());
                },
            },
        };

        qualname.push_str(get_ident(method.pe_ident()).get());
        let qualname = qualname.as_slice();

        // record the decl for this def (if it has one)
        let decl_id = ty::trait_item_of_item(&self.analysis.ty_cx,
                                             ast_util::local_def(method.id))
            .and_then(|def_id| {
                if match def_id {
                    ty::MethodTraitItemId(def_id) => {
                        method.id != 0 && def_id.node == 0
                    }
                    ty::TypeTraitItemId(_) => false,
                } {
                    Some(def_id)
                } else {
                    None
                }
            });
        let decl_id = match decl_id {
            None => None,
            Some(id) => Some(id.def_id()),
        };

        let sub_span = self.span.sub_span_after_keyword(method.span, keywords::Fn);
        self.fmt.method_str(method.span,
                            sub_span,
                            method.id,
                            qualname,
                            decl_id,
                            scope_id);

        self.process_formals(&method.pe_fn_decl().inputs, qualname);

        // walk arg and return types
        for arg in method.pe_fn_decl().inputs.iter() {
            self.visit_ty(&*arg.ty);
        }
        self.visit_ty(&*method.pe_fn_decl().output);
        // walk the fn body
        self.nest(method.id, |v| v.visit_block(&*method.pe_body()));

        self.process_generic_params(method.pe_generics(),
                                    method.span,
                                    qualname,
                                    method.id);
    }

    fn process_trait_ref(&mut self,
                         trait_ref: &ast::TraitRef,
                         impl_id: Option<NodeId>) {
        match self.lookup_type_ref(trait_ref.ref_id) {
            Some(id) => {
                let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                self.fmt.ref_str(recorder::TypeRef,
                                 trait_ref.path.span,
                                 sub_span,
                                 id,
                                 self.cur_scope);
                match impl_id {
                    Some(impl_id) => self.fmt.impl_str(trait_ref.path.span,
                                                       sub_span,
                                                       impl_id,
                                                       id,
                                                       self.cur_scope),
                    None => (),
                }
                visit::walk_path(self, &trait_ref.path);
            },
            None => ()
        }
    }

    fn process_struct_field_def(&mut self,
                                field: &ast::StructField,
                                qualname: &str,
                                scope_id: NodeId) {
        match field.node.kind {
            ast::NamedField(ident, _) => {
                let name = get_ident(ident);
                let qualname = format!("{}::{}", qualname, name);
                let typ = ppaux::ty_to_string(&self.analysis.ty_cx,
                    (*self.analysis.ty_cx.node_types.borrow())[field.node.id as uint]);
                match self.span.sub_span_before_token(field.span, token::COLON) {
                    Some(sub_span) => self.fmt.field_str(field.span,
                                                         Some(sub_span),
                                                         field.node.id,
                                                         name.get().as_slice(),
                                                         qualname.as_slice(),
                                                         typ.as_slice(),
                                                         scope_id),
                    None => self.sess.span_bug(field.span,
                                               format!("Could not find sub-span for field {}",
                                                       qualname).as_slice()),
                }
            },
            _ => (),
        }
    }

    // Dump generic params bindings, then visit_generics
    fn process_generic_params(&mut self, generics:&ast::Generics,
                              full_span: Span,
                              prefix: &str,
                              id: NodeId) {
        // We can't only use visit_generics since we don't have spans for param
        // bindings, so we reparse the full_span to get those sub spans.
        // However full span is the entire enum/fn/struct block, so we only want
        // the first few to match the number of generics we're looking for.
        let param_sub_spans = self.span.spans_for_ty_params(full_span,
                                                           (generics.ty_params.len() as int));
        for (param, param_ss) in generics.ty_params.iter().zip(param_sub_spans.iter()) {
            // Append $id to name to make sure each one is unique
            let name = format!("{}::{}${}",
                               prefix,
                               escape(self.span.snippet(*param_ss)),
                               id);
            self.fmt.typedef_str(full_span,
                                 Some(*param_ss),
                                 param.id,
                                 name.as_slice(),
                                 "");
        }
        self.visit_generics(generics);
    }

    fn process_fn(&mut self,
                  item: &ast::Item,
                  decl: &ast::FnDecl,
                  ty_params: &ast::Generics,
                  body: &ast::Block) {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Fn);
        self.fmt.fn_str(item.span,
                        sub_span,
                        item.id,
                        qualname.as_slice(),
                        self.cur_scope);

        self.process_formals(&decl.inputs, qualname.as_slice());

        // walk arg and return types
        for arg in decl.inputs.iter() {
            self.visit_ty(&*arg.ty);
        }
        self.visit_ty(&*decl.output);

        // walk the body
        self.nest(item.id, |v| v.visit_block(&*body));

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id);
    }

    fn process_static(&mut self,
                      item: &ast::Item,
                      typ: &ast::Ty,
                      mt: ast::Mutability,
                      expr: &ast::Expr)
    {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        // If the variable is immutable, save the initialising expression.
        let value = match mt {
            ast::MutMutable => String::from_str("<mutable>"),
            ast::MutImmutable => self.span.snippet(expr.span),
        };

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Static);
        self.fmt.static_str(item.span,
                            sub_span,
                            item.id,
                            get_ident(item.ident).get(),
                            qualname.as_slice(),
                            value.as_slice(),
                            ty_to_string(&*typ).as_slice(),
                            self.cur_scope);

        // walk type and init value
        self.visit_ty(&*typ);
        self.visit_expr(expr);
    }

    fn process_const(&mut self,
                      item: &ast::Item,
                      typ: &ast::Ty,
                      expr: &ast::Expr)
    {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        let sub_span = self.span.sub_span_after_keyword(item.span,
                                                        keywords::Const);
        self.fmt.static_str(item.span,
                            sub_span,
                            item.id,
                            get_ident(item.ident).get(),
                            qualname.as_slice(),
                            "",
                            ty_to_string(&*typ).as_slice(),
                            self.cur_scope);

        // walk type and init value
        self.visit_ty(&*typ);
        self.visit_expr(expr);
    }

    fn process_struct(&mut self,
                      item: &ast::Item,
                      def: &ast::StructDef,
                      ty_params: &ast::Generics) {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        let ctor_id = match def.ctor_id {
            Some(node_id) => node_id,
            None => -1,
        };
        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Struct);
        self.fmt.struct_str(item.span,
                            sub_span,
                            item.id,
                            ctor_id,
                            qualname.as_slice(),
                            self.cur_scope);

        // fields
        for field in def.fields.iter() {
            self.process_struct_field_def(field, qualname.as_slice(), item.id);
            self.visit_ty(&*field.node.ty);
        }

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id);
    }

    fn process_enum(&mut self,
                    item: &ast::Item,
                    enum_definition: &ast::EnumDef,
                    ty_params: &ast::Generics) {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);
        match self.span.sub_span_after_keyword(item.span, keywords::Enum) {
            Some(sub_span) => self.fmt.enum_str(item.span,
                                                Some(sub_span),
                                                item.id,
                                                qualname.as_slice(),
                                                self.cur_scope),
            None => self.sess.span_bug(item.span,
                                       format!("Could not find subspan for enum {}",
                                               qualname).as_slice()),
        }
        for variant in enum_definition.variants.iter() {
            let name = get_ident(variant.node.name);
            let name = name.get();
            let mut qualname = qualname.clone();
            qualname.push_str("::");
            qualname.push_str(name);
            let val = self.span.snippet(variant.span);
            match variant.node.kind {
                ast::TupleVariantKind(ref args) => {
                    // first ident in span is the variant's name
                    self.fmt.tuple_variant_str(variant.span,
                                               self.span.span_for_first_ident(variant.span),
                                               variant.node.id,
                                               name,
                                               qualname.as_slice(),
                                               val.as_slice(),
                                               item.id);
                    for arg in args.iter() {
                        self.visit_ty(&*arg.ty);
                    }
                }
                ast::StructVariantKind(ref struct_def) => {
                    let ctor_id = match struct_def.ctor_id {
                        Some(node_id) => node_id,
                        None => -1,
                    };
                    self.fmt.struct_variant_str(
                        variant.span,
                        self.span.span_for_first_ident(variant.span),
                        variant.node.id,
                        ctor_id,
                        qualname.as_slice(),
                        val.as_slice(),
                        item.id);

                    for field in struct_def.fields.iter() {
                        self.process_struct_field_def(field, qualname.as_slice(), variant.node.id);
                        self.visit_ty(&*field.node.ty);
                    }
                }
            }
        }

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id);
    }

    fn process_impl(&mut self,
                    item: &ast::Item,
                    type_parameters: &ast::Generics,
                    trait_ref: &Option<ast::TraitRef>,
                    typ: &ast::Ty,
                    impl_items: &Vec<ast::ImplItem>) {
        match typ.node {
            ast::TyPath(ref path, _, id) => {
                match self.lookup_type_ref(id) {
                    Some(id) => {
                        let sub_span = self.span.sub_span_for_type_name(path.span);
                        self.fmt.ref_str(recorder::TypeRef,
                                         path.span,
                                         sub_span,
                                         id,
                                         self.cur_scope);
                        self.fmt.impl_str(path.span,
                                          sub_span,
                                          item.id,
                                          id,
                                          self.cur_scope);
                    },
                    None => ()
                }
            },
            _ => self.visit_ty(&*typ),
        }

        match *trait_ref {
            Some(ref trait_ref) => self.process_trait_ref(trait_ref, Some(item.id)),
            None => (),
        }

        self.process_generic_params(type_parameters, item.span, "", item.id);
        for impl_item in impl_items.iter() {
            match *impl_item {
                ast::MethodImplItem(ref method) => {
                    visit::walk_method_helper(self, &**method)
                }
                ast::TypeImplItem(ref typedef) => {
                    visit::walk_ty(self, &*typedef.typ)
                }
            }
        }
    }

    fn process_trait(&mut self,
                     item: &ast::Item,
                     generics: &ast::Generics,
                     trait_refs: &OwnedSlice<ast::TyParamBound>,
                     methods: &Vec<ast::TraitItem>) {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Trait);
        self.fmt.trait_str(item.span,
                           sub_span,
                           item.id,
                           qualname.as_slice(),
                           self.cur_scope);

        // super-traits
        for super_bound in trait_refs.iter() {
            let trait_ref = match *super_bound {
                ast::TraitTyParamBound(ref trait_ref) => {
                    trait_ref
                }
                ast::UnboxedFnTyParamBound(..) | ast::RegionTyParamBound(..) => {
                    continue;
                }
            };

            match self.lookup_type_ref(trait_ref.ref_id) {
                Some(id) => {
                    let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                    self.fmt.ref_str(recorder::TypeRef,
                                     trait_ref.path.span,
                                     sub_span,
                                     id,
                                     self.cur_scope);
                    self.fmt.inherit_str(trait_ref.path.span,
                                         sub_span,
                                         id,
                                         item.id);
                },
                None => ()
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, item.span, qualname.as_slice(), item.id);
        for method in methods.iter() {
            self.visit_trait_item(method)
        }
    }

    fn process_mod(&mut self,
                   item: &ast::Item,  // The module in question, represented as an item.
                   m: &ast::Mod) {
        let qualname = self.analysis.ty_cx.map.path_to_string(item.id);

        let cm = self.sess.codemap();
        let filename = cm.span_to_filename(m.inner);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Mod);
        self.fmt.mod_str(item.span,
                         sub_span,
                         item.id,
                         qualname.as_slice(),
                         self.cur_scope,
                         filename.as_slice());

        self.nest(item.id, |v| visit::walk_mod(v, m));
    }

    fn process_path(&mut self,
                    ex: &ast::Expr,
                    path: &ast::Path) {
        if generated_code(path.span) {
            return
        }

        let def_map = self.analysis.ty_cx.def_map.borrow();
        if !def_map.contains_key(&ex.id) {
            self.sess.span_bug(ex.span,
                               format!("def_map has no key for {} in visit_expr",
                                       ex.id).as_slice());
        }
        let def = &(*def_map)[ex.id];
        let sub_span = self.span.span_for_last_ident(ex.span);
        match *def {
            def::DefUpvar(..) |
            def::DefLocal(..) |
            def::DefStatic(..) |
            def::DefConst(..) |
            def::DefVariant(..) => self.fmt.ref_str(recorder::VarRef,
                                                    ex.span,
                                                    sub_span,
                                                    def.def_id(),
                                                    self.cur_scope),
            def::DefStruct(def_id) => self.fmt.ref_str(recorder::StructRef,
                                                       ex.span,
                                                       sub_span,
                                                       def_id,
                                                        self.cur_scope),
            def::DefStaticMethod(declid, provenence, _) => {
                let sub_span = self.span.sub_span_for_meth_name(ex.span);
                let defid = if declid.krate == ast::LOCAL_CRATE {
                    let ti = ty::impl_or_trait_item(&self.analysis.ty_cx,
                                                    declid);
                    match provenence {
                        def::FromTrait(def_id) => {
                            Some(ty::trait_items(&self.analysis.ty_cx,
                                                 def_id)
                                    .iter()
                                    .find(|mr| {
                                        mr.ident().name == ti.ident().name
                                    })
                                    .unwrap()
                                    .def_id())
                        }
                        def::FromImpl(def_id) => {
                            let impl_items = self.analysis
                                                 .ty_cx
                                                 .impl_items
                                                 .borrow();
                            Some((*impl_items)[def_id]
                                           .iter()
                                           .find(|mr| {
                                            ty::impl_or_trait_item(
                                                &self.analysis.ty_cx,
                                                mr.def_id()).ident().name ==
                                                ti.ident().name
                                            })
                                           .unwrap()
                                           .def_id())
                        }
                    }
                } else {
                    None
                };
                self.fmt.meth_call_str(ex.span,
                                       sub_span,
                                       defid,
                                       Some(declid),
                                       self.cur_scope);
            },
            def::DefFn(def_id, _, _) => self.fmt.fn_call_str(ex.span,
                                                             sub_span,
                                                             def_id,
                                                             self.cur_scope),
            _ => self.sess.span_bug(ex.span,
                                    format!("Unexpected def kind while looking up path in '{}'",
                                            self.span.snippet(ex.span)).as_slice()),
        }
        // modules or types in the path prefix
        match *def {
            def::DefStaticMethod(_, _, _) => {
                self.write_sub_path_trait_truncated(path);
            },
            def::DefLocal(_) |
            def::DefStatic(_,_) |
            def::DefConst(..) |
            def::DefStruct(_) |
            def::DefFn(..) => self.write_sub_paths_truncated(path),
            _ => {},
        }

        visit::walk_path(self, path);
    }

    fn process_struct_lit(&mut self,
                          ex: &ast::Expr,
                          path: &ast::Path,
                          fields: &Vec<ast::Field>,
                          base: &Option<P<ast::Expr>>) {
        if generated_code(path.span) {
            return
        }

        let mut struct_def: Option<DefId> = None;
        match self.lookup_type_ref(ex.id) {
            Some(id) => {
                struct_def = Some(id);
                let sub_span = self.span.span_for_last_ident(path.span);
                self.fmt.ref_str(recorder::StructRef,
                                 path.span,
                                 sub_span,
                                 id,
                                 self.cur_scope);
            },
            None => ()
        }

        self.write_sub_paths_truncated(path);

        for field in fields.iter() {
            match struct_def {
                Some(struct_def) => {
                    let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, struct_def);
                    for f in fields.iter() {
                        if generated_code(field.ident.span) {
                            continue;
                        }
                        if f.name == field.ident.node.name {
                            // We don't really need a sub-span here, but no harm done
                            let sub_span = self.span.span_for_last_ident(field.ident.span);
                            self.fmt.ref_str(recorder::VarRef,
                                             field.ident.span,
                                             sub_span,
                                             f.id,
                                             self.cur_scope);
                        }
                    }
                }
                None => {}
            }

            self.visit_expr(&*field.expr)
        }
        visit::walk_expr_opt(self, base)
    }

    fn process_method_call(&mut self,
                           ex: &ast::Expr,
                           args: &Vec<P<ast::Expr>>) {
        let method_map = self.analysis.ty_cx.method_map.borrow();
        let method_callee = &(*method_map)[typeck::MethodCall::expr(ex.id)];
        let (def_id, decl_id) = match method_callee.origin {
            typeck::MethodStatic(def_id) |
            typeck::MethodStaticUnboxedClosure(def_id) => {
                // method invoked on an object with a concrete type (not a static method)
                let decl_id =
                    match ty::trait_item_of_item(&self.analysis.ty_cx,
                                                 def_id) {
                        None => None,
                        Some(decl_id) => Some(decl_id.def_id()),
                    };

                // This incantation is required if the method referenced is a
                // trait's default implementation.
                let def_id = match ty::impl_or_trait_item(&self.analysis
                                                               .ty_cx,
                                                          def_id) {
                    ty::MethodTraitItem(method) => {
                        method.provided_source.unwrap_or(def_id)
                    }
                    ty::TypeTraitItem(_) => def_id,
                };
                (Some(def_id), decl_id)
            }
            typeck::MethodTypeParam(ref mp) => {
                // method invoked on a type parameter
                let trait_item = ty::trait_item(&self.analysis.ty_cx,
                                                mp.trait_ref.def_id,
                                                mp.method_num);
                (None, Some(trait_item.def_id()))
            }
            typeck::MethodTraitObject(ref mo) => {
                // method invoked on a trait instance
                let trait_item = ty::trait_item(&self.analysis.ty_cx,
                                                mo.trait_ref.def_id,
                                                mo.method_num);
                (None, Some(trait_item.def_id()))
            }
        };
        let sub_span = self.span.sub_span_for_meth_name(ex.span);
        self.fmt.meth_call_str(ex.span,
                               sub_span,
                               def_id,
                               decl_id,
                               self.cur_scope);

        // walk receiver and args
        visit::walk_exprs(self, args.as_slice());
    }

    fn process_pat(&mut self, p:&ast::Pat) {
        if generated_code(p.span) {
            return
        }

        match p.node {
            ast::PatStruct(ref path, ref fields, _) => {
                self.collected_paths.push((p.id, path.clone(), false, recorder::StructRef));
                visit::walk_path(self, path);
                let struct_def = match self.lookup_type_ref(p.id) {
                    Some(sd) => sd,
                    None => {
                        self.sess.span_bug(p.span,
                                           format!("Could not find struct_def for `{}`",
                                                   self.span.snippet(p.span)).as_slice());
                    }
                };
                // The AST doesn't give us a span for the struct field, so we have
                // to figure out where it is by assuming it's the token before each colon.
                let field_spans = self.span.sub_spans_before_tokens(p.span,
                                                                    token::COMMA,
                                                                    token::COLON);
                if fields.len() != field_spans.len() {
                    self.sess.span_bug(p.span,
                        format!("Mismatched field count in '{}', found {}, expected {}",
                                self.span.snippet(p.span), field_spans.len(), fields.len()
                               ).as_slice());
                }
                for (field, &span) in fields.iter().zip(field_spans.iter()) {
                    self.visit_pat(&*field.pat);
                    if span.is_none() {
                        continue;
                    }
                    let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, struct_def);
                    for f in fields.iter() {
                        if f.name == field.ident.name {
                            self.fmt.ref_str(recorder::VarRef,
                                             p.span,
                                             span,
                                             f.id,
                                             self.cur_scope);
                            break;
                        }
                    }
                }
            }
            ast::PatEnum(ref path, _) => {
                self.collected_paths.push((p.id, path.clone(), false, recorder::VarRef));
                visit::walk_pat(self, p);
            }
            ast::PatIdent(bm, ref path1, ref optional_subpattern) => {
                let immut = match bm {
                    // Even if the ref is mut, you can't change the ref, only
                    // the data pointed at, so showing the initialising expression
                    // is still worthwhile.
                    ast::BindByRef(_) => true,
                    ast::BindByValue(mt) => {
                        match mt {
                            ast::MutMutable => false,
                            ast::MutImmutable => true,
                        }
                    }
                };
                // collect path for either visit_local or visit_arm
                let path = ast_util::ident_to_path(path1.span,path1.node);
                self.collected_paths.push((p.id, path, immut, recorder::VarRef));
                match *optional_subpattern {
                    None => {}
                    Some(ref subpattern) => self.visit_pat(&**subpattern)
                }
            }
            _ => visit::walk_pat(self, p)
        }
    }
}

impl<'l, 'tcx, 'v> Visitor<'v> for DxrVisitor<'l, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        if generated_code(item.span) {
            return
        }

        match item.node {
            ast::ItemFn(ref decl, _, _, ref ty_params, ref body) =>
                self.process_fn(item, &**decl, ty_params, &**body),
            ast::ItemStatic(ref typ, mt, ref expr) =>
                self.process_static(item, &**typ, mt, &**expr),
            ast::ItemConst(ref typ, ref expr) =>
                self.process_const(item, &**typ, &**expr),
            ast::ItemStruct(ref def, ref ty_params) => self.process_struct(item, &**def, ty_params),
            ast::ItemEnum(ref def, ref ty_params) => self.process_enum(item, def, ty_params),
            ast::ItemImpl(ref ty_params,
                          ref trait_ref,
                          ref typ,
                          ref impl_items) => {
                self.process_impl(item,
                                  ty_params,
                                  trait_ref,
                                  &**typ,
                                  impl_items)
            }
            ast::ItemTrait(ref generics, _, ref trait_refs, ref methods) =>
                self.process_trait(item, generics, trait_refs, methods),
            ast::ItemMod(ref m) => self.process_mod(item, m),
            ast::ItemTy(ref ty, ref ty_params) => {
                let qualname = self.analysis.ty_cx.map.path_to_string(item.id);
                let value = ty_to_string(&**ty);
                let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Type);
                self.fmt.typedef_str(item.span,
                                     sub_span,
                                     item.id,
                                     qualname.as_slice(),
                                     value.as_slice());

                self.visit_ty(&**ty);
                self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id);
            },
            ast::ItemMac(_) => (),
            _ => visit::walk_item(self, item),
        }
    }

    fn visit_generics(&mut self, generics: &ast::Generics) {
        for param in generics.ty_params.iter() {
            for bound in param.bounds.iter() {
                match *bound {
                    ast::TraitTyParamBound(ref trait_ref) => {
                        self.process_trait_ref(trait_ref, None);
                    }
                    _ => {}
                }
            }
            match param.default {
                Some(ref ty) => self.visit_ty(&**ty),
                None => {}
            }
        }
    }

    // We don't actually index functions here, that is done in visit_item/ItemFn.
    // Here we just visit methods.
    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _: NodeId) {
        if generated_code(s) {
            return;
        }

        match fk {
            visit::FkMethod(_, _, method) => self.process_method(method),
            _ => visit::walk_fn(self, fk, fd, b, s),
        }
    }

    fn visit_trait_item(&mut self, tm: &ast::TraitItem) {
        match *tm {
            ast::RequiredMethod(ref method_type) => {
                if generated_code(method_type.span) {
                    return;
                }

                let mut scope_id;
                let mut qualname = match ty::trait_of_item(&self.analysis.ty_cx,
                                                           ast_util::local_def(method_type.id)) {
                    Some(def_id) => {
                        scope_id = def_id.node;
                        let mut s = ty::item_path_str(&self.analysis.ty_cx, def_id);
                        s.push_str("::");
                        s
                    },
                    None => {
                        self.sess.span_bug(method_type.span,
                                           format!("Could not find trait for method {}",
                                                   method_type.id).as_slice());
                    },
                };

                qualname.push_str(get_ident(method_type.ident).get());
                let qualname = qualname.as_slice();

                let sub_span = self.span.sub_span_after_keyword(method_type.span, keywords::Fn);
                self.fmt.method_decl_str(method_type.span,
                                         sub_span,
                                         method_type.id,
                                         qualname,
                                         scope_id);

                // walk arg and return types
                for arg in method_type.decl.inputs.iter() {
                    self.visit_ty(&*arg.ty);
                }
                self.visit_ty(&*method_type.decl.output);

                self.process_generic_params(&method_type.generics,
                                            method_type.span,
                                            qualname,
                                            method_type.id);
            }
            ast::ProvidedMethod(ref method) => self.process_method(&**method),
            ast::TypeTraitItem(_) => {}
        }
    }

    fn visit_view_item(&mut self, i: &ast::ViewItem) {
        if generated_code(i.span) {
            return
        }

        match i.node {
            ast::ViewItemUse(ref path) => {
                match path.node {
                    ast::ViewPathSimple(ident, ref path, id) => {
                        let sub_span = self.span.span_for_last_ident(path.span);
                        let mod_id = match self.lookup_type_ref(id) {
                            Some(def_id) => {
                                match self.lookup_def_kind(id, path.span) {
                                    Some(kind) => self.fmt.ref_str(kind,
                                                                   path.span,
                                                                   sub_span,
                                                                   def_id,
                                                                   self.cur_scope),
                                    None => {},
                                }
                                Some(def_id)
                            },
                            None => None,
                        };

                        // 'use' always introduces an alias, if there is not an explicit
                        // one, there is an implicit one.
                        let sub_span =
                            match self.span.sub_span_before_token(path.span, token::EQ) {
                                Some(sub_span) => Some(sub_span),
                                None => sub_span,
                            };

                        self.fmt.use_alias_str(path.span,
                                               sub_span,
                                               id,
                                               mod_id,
                                               get_ident(ident).get(),
                                               self.cur_scope);
                        self.write_sub_paths_truncated(path);
                    }
                    ast::ViewPathGlob(ref path, _) => {
                        self.write_sub_paths(path);
                    }
                    ast::ViewPathList(ref path, ref list, _) => {
                        for plid in list.iter() {
                            match plid.node {
                                ast::PathListIdent { id, .. } => {
                                    match self.lookup_type_ref(id) {
                                        Some(def_id) =>
                                            match self.lookup_def_kind(id, plid.span) {
                                                Some(kind) => {
                                                    self.fmt.ref_str(
                                                        kind, plid.span,
                                                        Some(plid.span),
                                                        def_id, self.cur_scope);
                                                }
                                                None => ()
                                            },
                                        None => ()
                                    }
                                },
                                ast::PathListMod { .. } => ()
                            }
                        }

                        self.write_sub_paths(path);
                    }
                }
            },
            ast::ViewItemExternCrate(ident, ref s, id) => {
                let name = get_ident(ident);
                let name = name.get();
                let s = match *s {
                    Some((ref s, _)) => s.get().to_string(),
                    None => name.to_string(),
                };
                let sub_span = self.span.sub_span_after_keyword(i.span, keywords::Crate);
                let cnum = match self.sess.cstore.find_extern_mod_stmt_cnum(id) {
                    Some(cnum) => cnum,
                    None => 0,
                };
                self.fmt.extern_crate_str(i.span,
                                          sub_span,
                                          id,
                                          cnum,
                                          name,
                                          s.as_slice(),
                                          self.cur_scope);
            },
        }
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        if generated_code(t.span) {
            return
        }

        match t.node {
            ast::TyPath(ref path, _, id) => {
                match self.lookup_type_ref(id) {
                    Some(id) => {
                        let sub_span = self.span.sub_span_for_type_name(t.span);
                        self.fmt.ref_str(recorder::TypeRef,
                                         t.span,
                                         sub_span,
                                         id,
                                         self.cur_scope);
                    },
                    None => ()
                }

                self.write_sub_paths_truncated(path);

                visit::walk_path(self, path);
            },
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
            },
            ast::ExprPath(ref path) => self.process_path(ex, path),
            ast::ExprStruct(ref path, ref fields, ref base) =>
                self.process_struct_lit(ex, path, fields, base),
            ast::ExprMethodCall(_, _, ref args) => self.process_method_call(ex, args),
            ast::ExprField(ref sub_ex, ident, _) => {
                if generated_code(sub_ex.span) {
                    return
                }

                self.visit_expr(&**sub_ex);

                let t = ty::expr_ty_adjusted(&self.analysis.ty_cx, &**sub_ex);
                let t_box = ty::get(t);
                match t_box.sty {
                    ty::ty_struct(def_id, _) => {
                        let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, def_id);
                        for f in fields.iter() {
                            if f.name == ident.node.name {
                                let sub_span = self.span.span_for_last_ident(ex.span);
                                self.fmt.ref_str(recorder::VarRef,
                                                 ex.span,
                                                 sub_span,
                                                 f.id,
                                                 self.cur_scope);
                                break;
                            }
                        }
                    },
                    _ => self.sess.span_bug(ex.span,
                                            "Expected struct type, but not ty_struct"),
                }
            },
            ast::ExprTupField(ref sub_ex, idx, _) => {
                if generated_code(sub_ex.span) {
                    return
                }

                self.visit_expr(&**sub_ex);

                let t = ty::expr_ty_adjusted(&self.analysis.ty_cx, &**sub_ex);
                let t_box = ty::get(t);
                match t_box.sty {
                    ty::ty_struct(def_id, _) => {
                        let fields = ty::lookup_struct_fields(&self.analysis.ty_cx, def_id);
                        for (i, f) in fields.iter().enumerate() {
                            if i == idx.node {
                                let sub_span = self.span.span_for_last_ident(ex.span);
                                self.fmt.ref_str(recorder::VarRef,
                                                 ex.span,
                                                 sub_span,
                                                 f.id,
                                                 self.cur_scope);
                                break;
                            }
                        }
                    },
                    _ => self.sess.span_bug(ex.span,
                                            "Expected struct type, but not ty_struct"),
                }
            },
            ast::ExprFnBlock(_, ref decl, ref body) => {
                if generated_code(body.span) {
                    return
                }

                let mut id = String::from_str("$");
                id.push_str(ex.id.to_string().as_slice());
                self.process_formals(&decl.inputs, id.as_slice());

                // walk arg and return types
                for arg in decl.inputs.iter() {
                    self.visit_ty(&*arg.ty);
                }
                self.visit_ty(&*decl.output);

                // walk the body
                self.nest(ex.id, |v| v.visit_block(&**body));
            },
            _ => {
                visit::walk_expr(self, ex)
            },
        }
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // Just stop, macros are poison to us.
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        self.process_pat(p);
        if !self.collecting {
            self.collected_paths.clear();
        }
    }

    fn visit_arm(&mut self, arm: &ast::Arm) {
        assert!(self.collected_paths.len() == 0 && !self.collecting);
        self.collecting = true;

        for pattern in arm.pats.iter() {
            // collect paths from the arm's patterns
            self.visit_pat(&**pattern);
        }
        self.collecting = false;
        // process collected paths
        for &(id, ref p, ref immut, ref_kind) in self.collected_paths.iter() {
            let value = if *immut {
                self.span.snippet(p.span).into_string()
            } else {
                "<mutable>".to_string()
            };
            let sub_span = self.span.span_for_first_ident(p.span);
            let def_map = self.analysis.ty_cx.def_map.borrow();
            if !def_map.contains_key(&id) {
                self.sess.span_bug(p.span,
                                   format!("def_map has no key for {} in visit_arm",
                                           id).as_slice());
            }
            let def = &(*def_map)[id];
            match *def {
                def::DefLocal(id)  => self.fmt.variable_str(p.span,
                                                            sub_span,
                                                            id,
                                                            path_to_string(p).as_slice(),
                                                            value.as_slice(),
                                                            ""),
                def::DefVariant(_,id,_) => self.fmt.ref_str(ref_kind,
                                                            p.span,
                                                            sub_span,
                                                            id,
                                                            self.cur_scope),
                // FIXME(nrc) what is this doing here?
                def::DefStatic(_, _) => {}
                def::DefConst(..) => {}
                _ => error!("unexpected definition kind when processing collected paths: {}",
                            *def)
            }
        }
        self.collected_paths.clear();
        visit::walk_expr_opt(self, &arm.guard);
        self.visit_expr(&*arm.body);
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

        // The local could declare multiple new vars, we must walk the
        // pattern and collect them all.
        assert!(self.collected_paths.len() == 0 && !self.collecting);
        self.collecting = true;
        self.visit_pat(&*l.pat);
        self.collecting = false;

        let value = self.span.snippet(l.span);

        for &(id, ref p, ref immut, _) in self.collected_paths.iter() {
            let value = if *immut { value.to_string() } else { "<mutable>".to_string() };
            let types = self.analysis.ty_cx.node_types.borrow();
            let typ = ppaux::ty_to_string(&self.analysis.ty_cx, (*types)[id as uint]);
            // Get the span only for the name of the variable (I hope the path
            // is only ever a variable name, but who knows?).
            let sub_span = self.span.span_for_last_ident(p.span);
            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            self.fmt.variable_str(p.span,
                                  sub_span,
                                  id,
                                  path_to_string(p).as_slice(),
                                  value.as_slice(),
                                  typ.as_slice());
        }
        self.collected_paths.clear();

        // Just walk the initialiser and type (don't want to walk the pattern again).
        self.visit_ty(&*l.ty);
        visit::walk_expr_opt(self, &l.init);
    }
}

pub fn process_crate(sess: &Session,
                     krate: &ast::Crate,
                     analysis: &CrateAnalysis,
                     odir: &Option<Path>) {
    if generated_code(krate.span) {
        return;
    }

    let cratename = match attr::find_crate_name(krate.attrs.as_slice()) {
        Some(name) => name.get().to_string(),
        None => {
            info!("Could not find crate name, using 'unknown_crate'");
            String::from_str("unknown_crate")
        },
    };

    info!("Dumping crate {}", cratename);

    // find a path to dump our data to
    let mut root_path = match os::getenv("DXR_RUST_TEMP_FOLDER") {
        Some(val) => Path::new(val),
        None => match *odir {
            Some(ref val) => val.join("dxr"),
            None => Path::new("dxr-temp"),
        },
    };

    match fs::mkdir_recursive(&root_path, io::USER_RWX) {
        Err(e) => sess.err(format!("Could not create directory {}: {}",
                           root_path.display(), e).as_slice()),
        _ => (),
    }

    {
        let disp = root_path.display();
        info!("Writing output to {}", disp);
    }

    // Create output file.
    let mut out_name = cratename.clone();
    out_name.push_str(".csv");
    root_path.push(out_name);
    let output_file = match File::create(&root_path) {
        Ok(f) => box f,
        Err(e) => {
            let disp = root_path.display();
            sess.fatal(format!("Could not open {}: {}", disp, e).as_slice());
        }
    };
    root_path.pop();

    let mut visitor = DxrVisitor {
        sess: sess,
        analysis: analysis,
        collected_paths: vec!(),
        collecting: false,
        fmt: FmtStrs::new(box Recorder {
                            out: output_file as Box<Writer+'static>,
                            dump_spans: false,
                        },
                        SpanUtils {
                            sess: sess,
                            err_count: Cell::new(0)
                        },
                        cratename.clone()),
        span: SpanUtils {
            sess: sess,
            err_count: Cell::new(0)
        },
        cur_scope: 0
    };

    visitor.dump_crate_info(cratename.as_slice(), krate);

    visit::walk_crate(&mut visitor, krate);
}
