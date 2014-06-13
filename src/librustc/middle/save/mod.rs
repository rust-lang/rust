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
use std::gc::Gc;
use std::io;
use std::io::File;
use std::io::fs;
use std::os;

use syntax::ast;
use syntax::ast_util;
use syntax::ast::{NodeId,DefId};
use syntax::ast_map::NodeItem;
use syntax::attr;
use syntax::codemap::*;
use syntax::parse::token;
use syntax::parse::token::{get_ident,keywords};
use syntax::visit;
use syntax::visit::Visitor;
use syntax::print::pprust::{path_to_str,ty_to_str};

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
    span.expn_info.is_some() || span  == DUMMY_SP
}

struct DxrVisitor<'l> {
    sess: &'l Session,
    analysis: &'l CrateAnalysis,

    collected_paths: Vec<(NodeId, ast::Path, bool, recorder::Row)>,
    collecting: bool,

    span: SpanUtils<'l>,
    fmt: FmtStrs<'l>,
}

impl <'l> DxrVisitor<'l> {
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
                   path_to_str(path), spans.len(), path.segments.len());
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
            let qualname = path_to_str(&sub_path);
            result.push((*span, qualname));
            segs = sub_path.segments;
        }

        result
    }

    fn write_sub_paths(&mut self, path: &ast::Path, scope_id: NodeId) {
        let sub_paths = self.process_path_prefixes(path);
        for &(ref span, ref qualname) in sub_paths.iter() {
            self.fmt.sub_mod_ref_str(path.span,
                                     *span,
                                     qualname.as_slice(),
                                     scope_id);
        }
    }

    // As write_sub_paths, but does not process the last ident in the path (assuming it
    // will be processed elsewhere).
    fn write_sub_paths_truncated(&mut self, path: &ast::Path, scope_id: NodeId) {
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
                                     scope_id);
        }
    }

    // As write_sub_paths, but expects a path of the form module_path::trait::method
    // Where trait could actually be a struct too.
    fn write_sub_path_trait_truncated(&mut self, path: &ast::Path, scope_id: NodeId) {
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
        let sub_paths = sub_paths.slice(0, len-2);
        for &(ref span, ref qualname) in sub_paths.iter() {
            self.fmt.sub_mod_ref_str(path.span,
                                     *span,
                                     qualname.as_slice(),
                                     scope_id);
        }
    }

    // looks up anything, not just a type
    fn lookup_type_ref(&self, ref_id: NodeId) -> Option<DefId> {
        if !self.analysis.ty_cx.def_map.borrow().contains_key(&ref_id) {
            self.sess.bug(format!("def_map has no key for {} in lookup_type_ref",
                                  ref_id).as_slice());
        }
        let def = *self.analysis.ty_cx.def_map.borrow().get(&ref_id);
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
        let def = *def_map.get(&ref_id);
        match def {
            def::DefMod(_) |
            def::DefForeignMod(_) => Some(recorder::ModRef),
            def::DefStruct(_) => Some(recorder::StructRef),
            def::DefTy(_) |
            def::DefTrait(_) => Some(recorder::TypeRef),
            def::DefStatic(_, _) |
            def::DefBinding(_, _) |
            def::DefArg(_, _) |
            def::DefLocal(_, _) |
            def::DefVariant(_, _, _) |
            def::DefUpvar(_, _, _, _) => Some(recorder::VarRef),

            def::DefFn(_, _) => Some(recorder::FnRef),

            def::DefSelfTy(_) |
            def::DefRegion(_) |
            def::DefTyParamBinder(_) |
            def::DefLabel(_) |
            def::DefStaticMethod(_, _, _) |
            def::DefTyParam(..) |
            def::DefUse(_) |
            def::DefMethod(_, _) |
            def::DefPrimTy(_) => {
                self.sess.span_bug(span, format!("lookup_def_kind for unexpected item: {:?}",
                                                 def).as_slice());
            },
        }
    }

    fn process_formals(&mut self, formals: &Vec<ast::Arg>, qualname: &str, e:DxrVisitorEnv) {
        for arg in formals.iter() {
            assert!(self.collected_paths.len() == 0 && !self.collecting);
            self.collecting = true;
            self.visit_pat(&*arg.pat, e);
            self.collecting = false;
            let span_utils = self.span;
            for &(id, ref p, _, _) in self.collected_paths.iter() {
                let typ = ppaux::ty_to_str(&self.analysis.ty_cx,
                    *self.analysis.ty_cx.node_types.borrow().get(&(id as uint)));
                // get the span only for the name of the variable (I hope the path is only ever a
                // variable name, but who knows?)
                self.fmt.formal_str(p.span,
                                    span_utils.span_for_last_ident(p.span),
                                    id,
                                    qualname,
                                    path_to_str(p).as_slice(),
                                    typ.as_slice());
            }
            self.collected_paths.clear();
        }
    }

    fn process_method(&mut self, method: &ast::Method, e:DxrVisitorEnv) {
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
                        ast::ItemImpl(_, _, ty, _) => {
                            let mut result = String::from_str("<");
                            result.push_str(ty_to_str(&*ty).as_slice());

                            match ty::trait_of_method(&self.analysis.ty_cx,
                                                      ast_util::local_def(method.id)) {
                                Some(def_id) => {
                                    result.push_str(" as ");
                                    result.push_str(
                                        ty::item_path_str(&self.analysis.ty_cx, def_id).as_slice());
                                },
                                None => {}
                            }
                            result.append(">::")
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
                                       format!("Container {} for method {} is not a node item {:?}",
                                               impl_id.node,
                                               method.id,
                                               self.analysis.ty_cx.map.get(impl_id.node)
                                              ).as_slice());
                },
            },
            None => match ty::trait_of_method(&self.analysis.ty_cx,
                                              ast_util::local_def(method.id)) {
                Some(def_id) => {
                    scope_id = def_id.node;
                    match self.analysis.ty_cx.map.get(def_id.node) {
                        NodeItem(_) => {
                            let result = ty::item_path_str(&self.analysis.ty_cx, def_id);
                            result.append("::")
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

        qualname.push_str(get_ident(method.ident).get());
        let qualname = qualname.as_slice();

        // record the decl for this def (if it has one)
        let decl_id = ty::trait_method_of_method(&self.analysis.ty_cx,
                                                 ast_util::local_def(method.id))
            .filtered(|def_id| method.id != 0 && def_id.node == 0);

        let sub_span = self.span.sub_span_after_keyword(method.span, keywords::Fn);
        self.fmt.method_str(method.span,
                            sub_span,
                            method.id,
                            qualname,
                            decl_id,
                            scope_id);

        self.process_formals(&method.decl.inputs, qualname, e);

        // walk arg and return types
        for arg in method.decl.inputs.iter() {
            self.visit_ty(&*arg.ty, e);
        }
        self.visit_ty(&*method.decl.output, e);
        // walk the fn body
        self.visit_block(&*method.body, DxrVisitorEnv::new_nested(method.id));

        self.process_generic_params(&method.generics,
                                    method.span,
                                    qualname,
                                    method.id,
                                    e);
    }

    fn process_trait_ref(&mut self,
                         trait_ref: &ast::TraitRef,
                         e: DxrVisitorEnv,
                         impl_id: Option<NodeId>) {
        match self.lookup_type_ref(trait_ref.ref_id) {
            Some(id) => {
                let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                self.fmt.ref_str(recorder::TypeRef,
                                 trait_ref.path.span,
                                 sub_span,
                                 id,
                                 e.cur_scope);
                match impl_id {
                    Some(impl_id) => self.fmt.impl_str(trait_ref.path.span,
                                                       sub_span,
                                                       impl_id,
                                                       id,
                                                       e.cur_scope),
                    None => (),
                }
                visit::walk_path(self, &trait_ref.path, e);
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
                let typ = ppaux::ty_to_str(&self.analysis.ty_cx,
                    *self.analysis.ty_cx.node_types.borrow().get(&(field.node.id as uint)));
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
                              id: NodeId,
                              e: DxrVisitorEnv) {
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
        self.visit_generics(generics, e);
    }

    fn process_fn(&mut self,
                  item: &ast::Item,
                  e: DxrVisitorEnv,
                  decl: ast::P<ast::FnDecl>,
                  ty_params: &ast::Generics,
                  body: ast::P<ast::Block>) {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Fn);
        self.fmt.fn_str(item.span,
                        sub_span,
                        item.id,
                        qualname.as_slice(),
                        e.cur_scope);

        self.process_formals(&decl.inputs, qualname.as_slice(), e);

        // walk arg and return types
        for arg in decl.inputs.iter() {
            self.visit_ty(&*arg.ty, e);
        }
        self.visit_ty(&*decl.output, e);

        // walk the body
        self.visit_block(&*body, DxrVisitorEnv::new_nested(item.id));

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id, e);
    }

    fn process_static(&mut self,
                      item: &ast::Item,
                      e: DxrVisitorEnv,
                      typ: ast::P<ast::Ty>,
                      mt: ast::Mutability,
                      expr: &ast::Expr)
    {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);

        // If the variable is immutable, save the initialising expresion.
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
                            ty_to_str(&*typ).as_slice(),
                            e.cur_scope);

        // walk type and init value
        self.visit_ty(&*typ, e);
        self.visit_expr(expr, e);
    }

    fn process_struct(&mut self,
                      item: &ast::Item,
                      e: DxrVisitorEnv,
                      def: &ast::StructDef,
                      ty_params: &ast::Generics) {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);

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
                            e.cur_scope);

        // fields
        for field in def.fields.iter() {
            self.process_struct_field_def(field, qualname.as_slice(), item.id);
            self.visit_ty(&*field.node.ty, e);
        }

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id, e);
    }

    fn process_enum(&mut self,
                    item: &ast::Item,
                    e: DxrVisitorEnv,
                    enum_definition: &ast::EnumDef,
                    ty_params: &ast::Generics) {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);
        match self.span.sub_span_after_keyword(item.span, keywords::Enum) {
            Some(sub_span) => self.fmt.enum_str(item.span,
                                                Some(sub_span),
                                                item.id,
                                                qualname.as_slice(),
                                                e.cur_scope),
            None => self.sess.span_bug(item.span,
                                       format!("Could not find subspan for enum {}",
                                               qualname).as_slice()),
        }
        for variant in enum_definition.variants.iter() {
            let name = get_ident(variant.node.name);
            let name = name.get();
            let qualname = qualname.clone().append("::").append(name);
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
                        self.visit_ty(&*arg.ty, e);
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
                        self.visit_ty(&*field.node.ty, e);
                    }
                }
            }
        }

        self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id, e);
    }

    fn process_impl(&mut self,
                    item: &ast::Item,
                    e: DxrVisitorEnv,
                    type_parameters: &ast::Generics,
                    trait_ref: &Option<ast::TraitRef>,
                    typ: ast::P<ast::Ty>,
                    methods: &Vec<Gc<ast::Method>>) {
        match typ.node {
            ast::TyPath(ref path, _, id) => {
                match self.lookup_type_ref(id) {
                    Some(id) => {
                        let sub_span = self.span.sub_span_for_type_name(path.span);
                        self.fmt.ref_str(recorder::TypeRef,
                                         path.span,
                                         sub_span,
                                         id,
                                         e.cur_scope);
                        self.fmt.impl_str(path.span,
                                          sub_span,
                                          item.id,
                                          id,
                                          e.cur_scope);
                    },
                    None => ()
                }
            },
            _ => self.visit_ty(&*typ, e),
        }

        match *trait_ref {
            Some(ref trait_ref) => self.process_trait_ref(trait_ref, e, Some(item.id)),
            None => (),
        }

        self.process_generic_params(type_parameters, item.span, "", item.id, e);
        for method in methods.iter() {
            visit::walk_method_helper(self, &**method, e)
        }
    }

    fn process_trait(&mut self,
                     item: &ast::Item,
                     e: DxrVisitorEnv,
                     generics: &ast::Generics,
                     trait_refs: &Vec<ast::TraitRef>,
                     methods: &Vec<ast::TraitMethod>) {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Trait);
        self.fmt.trait_str(item.span,
                           sub_span,
                           item.id,
                           qualname.as_slice(),
                           e.cur_scope);

        // super-traits
        for trait_ref in trait_refs.iter() {
            match self.lookup_type_ref(trait_ref.ref_id) {
                Some(id) => {
                    let sub_span = self.span.sub_span_for_type_name(trait_ref.path.span);
                    self.fmt.ref_str(recorder::TypeRef,
                                     trait_ref.path.span,
                                     sub_span,
                                     id,
                                     e.cur_scope);
                    self.fmt.inherit_str(trait_ref.path.span,
                                         sub_span,
                                         id,
                                         item.id);
                },
                None => ()
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, item.span, qualname.as_slice(), item.id, e);
        for method in methods.iter() {
            self.visit_trait_method(method, e)
        }
    }

    fn process_mod(&mut self,
                   item: &ast::Item,  // The module in question, represented as an item.
                   e: DxrVisitorEnv,
                   m: &ast::Mod) {
        let qualname = self.analysis.ty_cx.map.path_to_str(item.id);

        let cm = self.sess.codemap();
        let filename = cm.span_to_filename(m.inner);

        let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Mod);
        self.fmt.mod_str(item.span,
                         sub_span,
                         item.id,
                         qualname.as_slice(),
                         e.cur_scope,
                         filename.as_slice());

        visit::walk_mod(self, m, DxrVisitorEnv::new_nested(item.id));
    }

    fn process_path(&mut self,
                    ex: &ast::Expr,
                    e: DxrVisitorEnv,
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
        let def = def_map.get(&ex.id);
        let sub_span = self.span.span_for_last_ident(ex.span);
        match *def {
            def::DefLocal(id, _) |
            def::DefArg(id, _) |
            def::DefUpvar(id, _, _, _) |
            def::DefBinding(id, _) => self.fmt.ref_str(recorder::VarRef,
                                                       ex.span,
                                                       sub_span,
                                                       ast_util::local_def(id),
                                                       e.cur_scope),
            def::DefStatic(def_id,_) |
            def::DefVariant(_, def_id, _) => self.fmt.ref_str(recorder::VarRef,
                                                              ex.span,
                                                              sub_span,
                                                              def_id,
                                                              e.cur_scope),
            def::DefStruct(def_id) => self.fmt.ref_str(recorder::StructRef,
                                                       ex.span,
                                                       sub_span,
                                                       def_id,
                                                        e.cur_scope),
            def::DefStaticMethod(declid, provenence, _) => {
                let sub_span = self.span.sub_span_for_meth_name(ex.span);
                let defid = if declid.krate == ast::LOCAL_CRATE {
                    let m = ty::method(&self.analysis.ty_cx, declid);
                    match provenence {
                        def::FromTrait(def_id) =>
                            Some(ty::trait_methods(&self.analysis.ty_cx, def_id)
                                .iter().find(|mr| mr.ident.name == m.ident.name).unwrap().def_id),
                        def::FromImpl(def_id) => {
                            let impl_methods = self.analysis.ty_cx.impl_methods.borrow();
                            Some(*impl_methods.get(&def_id)
                                .iter().find(|mr|
                                    ty::method(
                                        &self.analysis.ty_cx, **mr).ident.name == m.ident.name)
                                .unwrap())
                        }
                    }
                } else {
                    None
                };
                self.fmt.meth_call_str(ex.span,
                                       sub_span,
                                       defid,
                                       Some(declid),
                                       e.cur_scope);
            },
            def::DefFn(def_id, _) => self.fmt.fn_call_str(ex.span,
                                                          sub_span,
                                                          def_id,
                                                          e.cur_scope),
            _ => self.sess.span_bug(ex.span,
                                    format!("Unexpected def kind while looking up path in '{}'",
                                            self.span.snippet(ex.span)).as_slice()),
        }
        // modules or types in the path prefix
        match *def {
            def::DefStaticMethod(_, _, _) => {
                self.write_sub_path_trait_truncated(path, e.cur_scope);
            },
            def::DefLocal(_, _) |
            def::DefArg(_, _) |
            def::DefStatic(_,_) |
            def::DefStruct(_) |
            def::DefFn(_, _) => self.write_sub_paths_truncated(path, e.cur_scope),
            _ => {},
        }

        visit::walk_path(self, path, e);
    }

    fn process_struct_lit(&mut self,
                          ex: &ast::Expr,
                          e: DxrVisitorEnv,
                          path: &ast::Path,
                          fields: &Vec<ast::Field>,
                          base: Option<Gc<ast::Expr>>) {
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
                                 e.cur_scope);
            },
            None => ()
        }

        self.write_sub_paths_truncated(path, e.cur_scope);

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
                                             e.cur_scope);
                        }
                    }
                }
                None => {}
            }

            self.visit_expr(&*field.expr, e)
        }
        visit::walk_expr_opt(self, base, e)
    }

    fn process_method_call(&mut self,
                           ex: &ast::Expr,
                           e: DxrVisitorEnv,
                           args: &Vec<Gc<ast::Expr>>) {
        let method_map = self.analysis.ty_cx.method_map.borrow();
        let method_callee = method_map.get(&typeck::MethodCall::expr(ex.id));
        let (def_id, decl_id) = match method_callee.origin {
            typeck::MethodStatic(def_id) => {
                // method invoked on an object with a concrete type (not a static method)
                let decl_id = ty::trait_method_of_method(&self.analysis.ty_cx, def_id);

                // This incantation is required if the method referenced is a trait's
                // defailt implementation.
                let def_id = ty::method(&self.analysis.ty_cx, def_id).provided_source
                                    .unwrap_or(def_id);
                (Some(def_id), decl_id)
            }
            typeck::MethodParam(mp) => {
                // method invoked on a type parameter
                let method = ty::trait_method(&self.analysis.ty_cx,
                                              mp.trait_id,
                                              mp.method_num);
                (None, Some(method.def_id))
            },
            typeck::MethodObject(mo) => {
                // method invoked on a trait instance
                let method = ty::trait_method(&self.analysis.ty_cx,
                                              mo.trait_id,
                                              mo.method_num);
                (None, Some(method.def_id))
            },
        };
        let sub_span = self.span.sub_span_for_meth_name(ex.span);
        self.fmt.meth_call_str(ex.span,
                               sub_span,
                               def_id,
                               decl_id,
                               e.cur_scope);

        // walk receiver and args
        visit::walk_exprs(self, args.as_slice(), e);
    }

    fn process_pat(&mut self, p:&ast::Pat, e: DxrVisitorEnv) {
        if generated_code(p.span) {
            return
        }

        match p.node {
            ast::PatStruct(ref path, ref fields, _) => {
                self.collected_paths.push((p.id, path.clone(), false, recorder::StructRef));
                visit::walk_path(self, path, e);
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
                    self.visit_pat(&*field.pat, e);
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
                                             e.cur_scope);
                            break;
                        }
                    }
                }
            }
            ast::PatEnum(ref path, _) => {
                self.collected_paths.push((p.id, path.clone(), false, recorder::VarRef));
                visit::walk_pat(self, p, e);
            }
            ast::PatIdent(bm, ref path, ref optional_subpattern) => {
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
                self.collected_paths.push((p.id, path.clone(), immut, recorder::VarRef));
                match *optional_subpattern {
                    None => {}
                    Some(subpattern) => self.visit_pat(&*subpattern, e),
                }
            }
            _ => visit::walk_pat(self, p, e)
        }
    }
}

impl<'l> Visitor<DxrVisitorEnv> for DxrVisitor<'l> {
    fn visit_item(&mut self, item:&ast::Item, e: DxrVisitorEnv) {
        if generated_code(item.span) {
            return
        }

        match item.node {
            ast::ItemFn(decl, _, _, ref ty_params, body) =>
                self.process_fn(item, e, decl, ty_params, body),
            ast::ItemStatic(typ, mt, expr) =>
                self.process_static(item, e, typ, mt, &*expr),
            ast::ItemStruct(def, ref ty_params) => self.process_struct(item, e, &*def, ty_params),
            ast::ItemEnum(ref def, ref ty_params) => self.process_enum(item, e, def, ty_params),
            ast::ItemImpl(ref ty_params, ref trait_ref, typ, ref methods) =>
                self.process_impl(item, e, ty_params, trait_ref, typ, methods),
            ast::ItemTrait(ref generics, _, ref trait_refs, ref methods) =>
                self.process_trait(item, e, generics, trait_refs, methods),
            ast::ItemMod(ref m) => self.process_mod(item, e, m),
            ast::ItemTy(ty, ref ty_params) => {
                let qualname = self.analysis.ty_cx.map.path_to_str(item.id);
                let value = ty_to_str(&*ty);
                let sub_span = self.span.sub_span_after_keyword(item.span, keywords::Type);
                self.fmt.typedef_str(item.span,
                                     sub_span,
                                     item.id,
                                     qualname.as_slice(),
                                     value.as_slice());

                self.visit_ty(&*ty, e);
                self.process_generic_params(ty_params, item.span, qualname.as_slice(), item.id, e);
            },
            ast::ItemMac(_) => (),
            _ => visit::walk_item(self, item, e),
        }
    }

    fn visit_generics(&mut self, generics: &ast::Generics, e: DxrVisitorEnv) {
        for param in generics.ty_params.iter() {
            for bound in param.bounds.iter() {
                match *bound {
                    ast::TraitTyParamBound(ref trait_ref) => {
                        self.process_trait_ref(trait_ref, e, None);
                    }
                    _ => {}
                }
            }
            match param.default {
                Some(ty) => self.visit_ty(&*ty, e),
                None => (),
            }
        }
    }

    // We don't actually index functions here, that is done in visit_item/ItemFn.
    // Here we just visit methods.
    fn visit_fn(&mut self,
                fk: &visit::FnKind,
                fd: &ast::FnDecl,
                b: &ast::Block,
                s: Span,
                _: NodeId,
                e: DxrVisitorEnv) {
        if generated_code(s) {
            return;
        }

        match *fk {
            visit::FkMethod(_, _, method) => self.process_method(method, e),
            _ => visit::walk_fn(self, fk, fd, b, s, e),
        }
    }

    fn visit_trait_method(&mut self, tm: &ast::TraitMethod, e: DxrVisitorEnv) {
        match *tm {
            ast::Required(ref method_type) => {
                if generated_code(method_type.span) {
                    return;
                }

                let mut scope_id ;
                let mut qualname = match ty::trait_of_method(&self.analysis.ty_cx,
                                                             ast_util::local_def(method_type.id)) {
                    Some(def_id) => {
                        scope_id = def_id.node;
                        ty::item_path_str(&self.analysis.ty_cx, def_id).append("::")
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
                    self.visit_ty(&*arg.ty, e);
                }
                self.visit_ty(&*method_type.decl.output, e);

                self.process_generic_params(&method_type.generics,
                                            method_type.span,
                                            qualname,
                                            method_type.id,
                                            e);
            }
            ast::Provided(method) => self.process_method(&*method, e),
        }
    }

    fn visit_view_item(&mut self, i:&ast::ViewItem, e:DxrVisitorEnv) {
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
                                                                   e.cur_scope),
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
                                               e.cur_scope);
                        self.write_sub_paths_truncated(path, e.cur_scope);
                    }
                    ast::ViewPathGlob(ref path, _) => {
                        self.write_sub_paths(path, e.cur_scope);
                    }
                    ast::ViewPathList(ref path, ref list, _) => {
                        for plid in list.iter() {
                            match self.lookup_type_ref(plid.node.id) {
                                Some(id) => match self.lookup_def_kind(plid.node.id, plid.span) {
                                    Some(kind) => self.fmt.ref_str(kind,
                                                                   plid.span,
                                                                   Some(plid.span),
                                                                   id,
                                                                   e.cur_scope),
                                    None => (),
                                },
                                None => ()
                            }
                        }

                        self.write_sub_paths(path, e.cur_scope);
                    }
                }
            },
            ast::ViewItemExternCrate(ident, ref s, id) => {
                let name = get_ident(ident).get().to_owned();
                let s = match *s {
                    Some((ref s, _)) => s.get().to_owned(),
                    None => name.to_owned(),
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
                                          name.as_slice(),
                                          s.as_slice(),
                                          e.cur_scope);
            },
        }
    }

    fn visit_ty(&mut self, t: &ast::Ty, e: DxrVisitorEnv) {
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
                                         e.cur_scope);
                    },
                    None => ()
                }

                self.write_sub_paths_truncated(path, e.cur_scope);

                visit::walk_path(self, path, e);
            },
            _ => visit::walk_ty(self, t, e),
        }
    }

    fn visit_expr(&mut self, ex: &ast::Expr, e: DxrVisitorEnv) {
        if generated_code(ex.span) {
            return
        }

        match ex.node {
            ast::ExprCall(_f, ref _args) => {
                // Don't need to do anything for function calls,
                // because just walking the callee path does what we want.
                visit::walk_expr(self, ex, e);
            },
            ast::ExprPath(ref path) => self.process_path(ex, e, path),
            ast::ExprStruct(ref path, ref fields, base) =>
                self.process_struct_lit(ex, e, path, fields, base),
            ast::ExprMethodCall(_, _, ref args) => self.process_method_call(ex, e, args),
            ast::ExprField(sub_ex, ident, _) => {
                if generated_code(sub_ex.span) {
                    return
                }

                self.visit_expr(&*sub_ex, e);

                let t = ty::expr_ty_adjusted(&self.analysis.ty_cx, &*sub_ex);
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
                                                 e.cur_scope);
                                break;
                            }
                        }
                    },
                    _ => self.sess.span_bug(ex.span,
                                            "Expected struct type, but not ty_struct"),
                }
            },
            ast::ExprFnBlock(decl, body) => {
                if generated_code(body.span) {
                    return
                }

                let id = String::from_str("$").append(ex.id.to_str().as_slice());
                self.process_formals(&decl.inputs, id.as_slice(), e);

                // walk arg and return types
                for arg in decl.inputs.iter() {
                    self.visit_ty(&*arg.ty, e);
                }
                self.visit_ty(&*decl.output, e);

                // walk the body
                self.visit_block(&*body, DxrVisitorEnv::new_nested(ex.id));
            },
            _ => {
                visit::walk_expr(self, ex, e)
            },
        }
    }

    fn visit_mac(&mut self, _: &ast::Mac, _: DxrVisitorEnv) {
        // Just stop, macros are poison to us.
    }

    fn visit_pat(&mut self, p: &ast::Pat, e: DxrVisitorEnv) {
        self.process_pat(p, e);
        if !self.collecting {
            self.collected_paths.clear();
        }
    }

    fn visit_arm(&mut self, arm: &ast::Arm, e: DxrVisitorEnv) {
        assert!(self.collected_paths.len() == 0 && !self.collecting);
        self.collecting = true;

        for pattern in arm.pats.iter() {
            // collect paths from the arm's patterns
            self.visit_pat(&**pattern, e);
        }
        self.collecting = false;
        // process collected paths
        for &(id, ref p, ref immut, ref_kind) in self.collected_paths.iter() {
            let value = if *immut {
                self.span.snippet(p.span).into_owned()
            } else {
                "<mutable>".to_owned()
            };
            let sub_span = self.span.span_for_first_ident(p.span);
            let def_map = self.analysis.ty_cx.def_map.borrow();
            if !def_map.contains_key(&id) {
                self.sess.span_bug(p.span,
                                   format!("def_map has no key for {} in visit_arm",
                                           id).as_slice());
            }
            let def = def_map.get(&id);
            match *def {
                def::DefBinding(id, _)  => self.fmt.variable_str(p.span,
                                                                 sub_span,
                                                                 id,
                                                                 path_to_str(p).as_slice(),
                                                                 value.as_slice(),
                                                                 ""),
                def::DefVariant(_,id,_) => self.fmt.ref_str(ref_kind,
                                                            p.span,
                                                            sub_span,
                                                            id,
                                                            e.cur_scope),
                // FIXME(nrc) what is this doing here?
                def::DefStatic(_, _) => {}
                _ => error!("unexpected defintion kind when processing collected paths: {:?}", *def)
            }
        }
        self.collected_paths.clear();
        visit::walk_expr_opt(self, arm.guard, e);
        self.visit_expr(&*arm.body, e);
    }

    fn visit_stmt(&mut self, s:&ast::Stmt, e:DxrVisitorEnv) {
        if generated_code(s.span) {
            return
        }

        visit::walk_stmt(self, s, e)
    }

    fn visit_local(&mut self, l:&ast::Local, e: DxrVisitorEnv) {
        if generated_code(l.span) {
            return
        }

        // The local could declare multiple new vars, we must walk the
        // pattern and collect them all.
        assert!(self.collected_paths.len() == 0 && !self.collecting);
        self.collecting = true;
        self.visit_pat(&*l.pat, e);
        self.collecting = false;

        let value = self.span.snippet(l.span);

        for &(id, ref p, ref immut, _) in self.collected_paths.iter() {
            let value = if *immut { value.to_owned() } else { "<mutable>".to_owned() };
            let types = self.analysis.ty_cx.node_types.borrow();
            let typ = ppaux::ty_to_str(&self.analysis.ty_cx, *types.get(&(id as uint)));
            // Get the span only for the name of the variable (I hope the path
            // is only ever a variable name, but who knows?).
            let sub_span = self.span.span_for_last_ident(p.span);
            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            self.fmt.variable_str(p.span,
                                  sub_span,
                                  id,
                                  path_to_str(p).as_slice(),
                                  value.as_slice(),
                                  typ.as_slice());
        }
        self.collected_paths.clear();

        // Just walk the initialiser and type (don't want to walk the pattern again).
        self.visit_ty(&*l.ty, e);
        visit::walk_expr_opt(self, l.init, e);
    }
}

#[deriving(Clone)]
struct DxrVisitorEnv {
    cur_scope: NodeId,
}

impl DxrVisitorEnv {
    fn new() -> DxrVisitorEnv {
        DxrVisitorEnv{cur_scope: 0}
    }
    fn new_nested(new_mod: NodeId) -> DxrVisitorEnv {
        DxrVisitorEnv{cur_scope: new_mod}
    }
}

pub fn process_crate(sess: &Session,
                     krate: &ast::Crate,
                     analysis: &CrateAnalysis,
                     odir: &Option<Path>) {
    if generated_code(krate.span) {
        return;
    }

    let (cratename, crateid) = match attr::find_crateid(krate.attrs.as_slice()) {
        Some(crateid) => (crateid.name.clone(), crateid.to_str()),
        None => {
            info!("Could not find crate name, using 'unknown_crate'");
            (String::from_str("unknown_crate"),"unknown_crate".to_owned())
        },
    };

    info!("Dumping crate {} ({})", cratename, crateid);

    // find a path to dump our data to
    let mut root_path = match os::getenv("DXR_RUST_TEMP_FOLDER") {
        Some(val) => Path::new(val),
        None => match *odir {
            Some(ref val) => val.join("dxr"),
            None => Path::new("dxr-temp"),
        },
    };

    match fs::mkdir_recursive(&root_path, io::UserRWX) {
        Err(e) => sess.err(format!("Could not create directory {}: {}",
                           root_path.display(), e).as_slice()),
        _ => (),
    }

    {
        let disp = root_path.display();
        info!("Writing output to {}", disp);
    }

    // Create ouput file.
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

    let mut visitor = DxrVisitor{ sess: sess,
                                  analysis: analysis,
                                  collected_paths: vec!(),
                                  collecting: false,
                                  fmt: FmtStrs::new(box Recorder {
                                                        out: output_file as Box<Writer>,
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
                                  }};

    visitor.dump_crate_info(cratename.as_slice(), krate);

    visit::walk_crate(&mut visitor, krate, DxrVisitorEnv::new());
}
