// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Feature gating
//!
//! This modules implements the gating necessary for preventing certain compiler
//! features from being used by default. This module will crawl a pre-expanded
//! AST to ensure that there are no features which are used that are not
//! enabled.
//!
//! Features are enabled in programs via the crate-level attributes of
//! #![feature(...)] with a comma-separated list of features.

use middle::lint;

use syntax::ast;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::parse::token;

use driver::session::Session;

use std::cell::Cell;

/// This is a list of all known features since the beginning of time. This list
/// can never shrink, it may only be expanded (in order to prevent old programs
/// from failing to compile). The status of each feature may change, however.
static KNOWN_FEATURES: &'static [(&'static str, Status)] = &[
    ("globs", Active),
    ("macro_rules", Active),
    ("struct_variant", Active),
    ("once_fns", Active),
    ("asm", Active),
    ("managed_boxes", Active),
    ("non_ascii_idents", Active),
    ("thread_local", Active),
    ("link_args", Active),
    ("phase", Active),
    ("plugin_registrar", Active),
    ("log_syntax", Active),
    ("trace_macros", Active),
    ("concat_idents", Active),

    ("simd", Active),
    ("default_type_params", Active),
    ("quote", Active),
    ("linkage", Active),
    ("struct_inherit", Active),
    ("overloaded_calls", Active),

    ("quad_precision_float", Active),

    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    ("issue_5723_bootstrap", Active),

    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    ("test_accepted_feature", Accepted),
    ("test_removed_feature", Removed),
];

enum Status {
    /// Represents an active feature that is currently being implemented or
    /// currently being considered for addition/removal.
    Active,

    /// Represents a feature which has since been removed (it was once Active)
    Removed,

    /// This language feature has since been Accepted (it was once Active)
    Accepted,
}

/// A set of features to be used by later passes.
pub struct Features {
    pub default_type_params: Cell<bool>,
    pub quad_precision_float: Cell<bool>,
    pub issue_5723_bootstrap: Cell<bool>,
    pub overloaded_calls: Cell<bool>,
}

impl Features {
    pub fn new() -> Features {
        Features {
            default_type_params: Cell::new(false),
            quad_precision_float: Cell::new(false),
            issue_5723_bootstrap: Cell::new(false),
            overloaded_calls: Cell::new(false),
        }
    }
}

struct Context<'a> {
    features: Vec<&'static str>,
    sess: &'a Session,
}

impl<'a> Context<'a> {
    fn gate_feature(&self, feature: &str, span: Span, explain: &str) {
        if !self.has_feature(feature) {
            self.sess.span_err(span, explain);
            self.sess.span_note(span, format!("add \\#![feature({})] to the \
                                                  crate attributes to enable",
                                                 feature).as_slice());
        }
    }

    fn gate_box(&self, span: Span) {
        self.gate_feature("managed_boxes", span,
                          "The managed box syntax is being replaced by the \
                           `std::gc::Gc` and `std::rc::Rc` types. Equivalent \
                           functionality to managed trait objects will be \
                           implemented but is currently missing.");
    }

    fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|n| n.as_slice() == feature)
    }
}

impl<'a> Visitor<()> for Context<'a> {
    fn visit_ident(&mut self, sp: Span, id: ast::Ident, _: ()) {
        if !token::get_ident(id).get().is_ascii() {
            self.gate_feature("non_ascii_idents", sp,
                              "non-ascii idents are not fully supported.");
        }
    }

    fn visit_view_item(&mut self, i: &ast::ViewItem, _: ()) {
        match i.node {
            ast::ViewItemUse(ref path) => {
                match path.node {
                    ast::ViewPathGlob(..) => {
                        self.gate_feature("globs", path.span,
                                          "glob import statements are \
                                           experimental and possibly buggy");
                    }
                    _ => {}
                }
            }
            ast::ViewItemExternCrate(..) => {
                for attr in i.attrs.iter() {
                    if attr.name().get() == "phase"{
                        self.gate_feature("phase", attr.span,
                                          "compile time crate loading is \
                                           experimental and possibly buggy");
                    }
                }
            }
        }
        visit::walk_view_item(self, i, ())
    }

    fn visit_item(&mut self, i: &ast::Item, _:()) {
        for attr in i.attrs.iter() {
            if attr.name().equiv(&("thread_local")) {
                self.gate_feature("thread_local", i.span,
                                  "`#[thread_local]` is an experimental feature, and does not \
                                  currently handle destructors. There is no corresponding \
                                  `#[task_local]` mapping to the task model");
            }
        }
        match i.node {
            ast::ItemEnum(ref def, _) => {
                for variant in def.variants.iter() {
                    match variant.node.kind {
                        ast::StructVariantKind(..) => {
                            self.gate_feature("struct_variant", variant.span,
                                              "enum struct variants are \
                                               experimental and possibly buggy");
                        }
                        _ => {}
                    }
                }
            }

            ast::ItemForeignMod(..) => {
                if attr::contains_name(i.attrs.as_slice(), "link_args") {
                    self.gate_feature("link_args", i.span,
                                      "the `link_args` attribute is not portable \
                                       across platforms, it is recommended to \
                                       use `#[link(name = \"foo\")]` instead")
                }
            }

            ast::ItemFn(..) => {
                if attr::contains_name(i.attrs.as_slice(), "plugin_registrar") {
                    self.gate_feature("plugin_registrar", i.span,
                                      "compiler plugins are experimental and possibly buggy");
                }
            }

            ast::ItemStruct(struct_definition, _) => {
                if attr::contains_name(i.attrs.as_slice(), "simd") {
                    self.gate_feature("simd", i.span,
                                      "SIMD types are experimental and possibly buggy");
                }
                match struct_definition.super_struct {
                    Some(ref path) => self.gate_feature("struct_inherit", path.span,
                                                        "struct inheritance is experimental \
                                                         and possibly buggy"),
                    None => {}
                }
                if struct_definition.is_virtual {
                    self.gate_feature("struct_inherit", i.span,
                                      "struct inheritance (`virtual` keyword) is \
                                       experimental and possibly buggy");
                }
            }

            _ => {}
        }

        visit::walk_item(self, i, ());
    }

    fn visit_mac(&mut self, macro: &ast::Mac, _: ()) {
        let ast::MacInvocTT(ref path, _, _) = macro.node;
        let id = path.segments.last().unwrap().identifier;
        let quotes = ["quote_tokens", "quote_expr", "quote_ty",
                      "quote_item", "quote_pat", "quote_stmt"];
        let msg = " is not stable enough for use and are subject to change";


        if id == token::str_to_ident("macro_rules") {
            self.gate_feature("macro_rules", path.span, "macro definitions are \
                not stable enough for use and are subject to change");
        }

        else if id == token::str_to_ident("asm") {
            self.gate_feature("asm", path.span, "inline assembly is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("log_syntax") {
            self.gate_feature("log_syntax", path.span, "`log_syntax!` is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("trace_macros") {
            self.gate_feature("trace_macros", path.span, "`trace_macros` is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("concat_idents") {
            self.gate_feature("concat_idents", path.span, "`concat_idents` is not \
                stable enough for use and is subject to change");
        }

        else {
            for &quote in quotes.iter() {
                if id == token::str_to_ident(quote) {
                  self.gate_feature("quote",
                                    path.span,
                                    format!("{}{}", quote, msg).as_slice());
                }
            }
        }
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem, _: ()) {
        match i.node {
            ast::ForeignItemFn(..) | ast::ForeignItemStatic(..) => {
                if attr::contains_name(i.attrs.as_slice(), "linkage") {
                    self.gate_feature("linkage", i.span,
                                      "the `linkage` attribute is experimental \
                                       and not portable across platforms")
                }
            }
        }
        visit::walk_foreign_item(self, i, ())
    }

    fn visit_ty(&mut self, t: &ast::Ty, _: ()) {
        match t.node {
            ast::TyClosure(closure, _) if closure.onceness == ast::Once => {
                self.gate_feature("once_fns", t.span,
                                  "once functions are \
                                   experimental and likely to be removed");

            },
            ast::TyBox(_) => { self.gate_box(t.span); }
            _ => {}
        }

        visit::walk_ty(self, t, ());
    }

    fn visit_expr(&mut self, e: &ast::Expr, _: ()) {
        match e.node {
            ast::ExprUnary(ast::UnBox, _) => {
                self.gate_box(e.span);
            }
            _ => {}
        }
        visit::walk_expr(self, e, ());
    }

    fn visit_generics(&mut self, generics: &ast::Generics, _: ()) {
        for type_parameter in generics.ty_params.iter() {
            match type_parameter.default {
                Some(ty) => {
                    self.gate_feature("default_type_params", ty.span,
                                      "default type parameters are \
                                       experimental and possibly buggy");
                }
                None => {}
            }
        }
        visit::walk_generics(self, generics, ());
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    let mut cx = Context {
        features: Vec::new(),
        sess: sess,
    };

    for attr in krate.attrs.iter() {
        if !attr.check_name("feature") {
            continue
        }

        match attr.meta_item_list() {
            None => {
                sess.span_err(attr.span, "malformed feature attribute, \
                                          expected #![feature(...)]");
            }
            Some(list) => {
                for &mi in list.iter() {
                    let name = match mi.node {
                        ast::MetaWord(ref word) => (*word).clone(),
                        _ => {
                            sess.span_err(mi.span,
                                          "malformed feature, expected just \
                                           one word");
                            continue
                        }
                    };
                    match KNOWN_FEATURES.iter()
                                        .find(|& &(n, _)| name.equiv(&n)) {
                        Some(&(name, Active)) => { cx.features.push(name); }
                        Some(&(_, Removed)) => {
                            sess.span_err(mi.span, "feature has been removed");
                        }
                        Some(&(_, Accepted)) => {
                            sess.span_warn(mi.span, "feature has added to rust, \
                                                     directive not necessary");
                        }
                        None => {
                            sess.add_lint(lint::UnknownFeatures,
                                          ast::CRATE_NODE_ID,
                                          mi.span,
                                          "unknown feature".to_string());
                        }
                    }
                }
            }
        }
    }

    visit::walk_crate(&mut cx, krate, ());

    sess.abort_if_errors();

    sess.features.default_type_params.set(cx.has_feature("default_type_params"));
    sess.features.quad_precision_float.set(cx.has_feature("quad_precision_float"));
    sess.features.issue_5723_bootstrap.set(cx.has_feature("issue_5723_bootstrap"));
    sess.features.overloaded_calls.set(cx.has_feature("overloaded_calls"));
}
