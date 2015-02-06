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
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once code for detection of feature
//! gate usage is added, *do not remove it again* even once the feature
//! becomes stable.
use self::Status::*;

use abi::RustIntrinsic;
use ast::NodeId;
use ast;
use attr;
use attr::AttrMetaMethods;
use codemap::{CodeMap, Span};
use diagnostic::SpanHandler;
use visit;
use visit::Visitor;
use parse::token::{self, InternedString};

use std::slice;
use std::ascii::AsciiExt;

// If you change this list without updating src/doc/reference.md, @cmr will be sad
// Don't ever remove anything from this list; set them to 'Removed'.
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
// NB: The featureck.py script parses this information directly out of the source
// so take care when modifying it.
static KNOWN_FEATURES: &'static [(&'static str, &'static str, Status)] = &[
    ("globs", "1.0.0", Accepted),
    ("macro_rules", "1.0.0", Accepted),
    ("struct_variant", "1.0.0", Accepted),
    ("asm", "1.0.0", Active),
    ("managed_boxes", "1.0.0", Removed),
    ("non_ascii_idents", "1.0.0", Active),
    ("thread_local", "1.0.0", Active),
    ("link_args", "1.0.0", Active),
    ("phase", "1.0.0", Removed),
    ("plugin_registrar", "1.0.0", Active),
    ("log_syntax", "1.0.0", Active),
    ("trace_macros", "1.0.0", Active),
    ("concat_idents", "1.0.0", Active),
    ("unsafe_destructor", "1.0.0", Active),
    ("intrinsics", "1.0.0", Active),
    ("lang_items", "1.0.0", Active),

    ("simd", "1.0.0", Active),
    ("default_type_params", "1.0.0", Accepted),
    ("quote", "1.0.0", Active),
    ("link_llvm_intrinsics", "1.0.0", Active),
    ("linkage", "1.0.0", Active),
    ("struct_inherit", "1.0.0", Removed),

    ("quad_precision_float", "1.0.0", Removed),

    ("rustc_diagnostic_macros", "1.0.0", Active),
    ("unboxed_closures", "1.0.0", Active),
    ("import_shadowing", "1.0.0", Removed),
    ("advanced_slice_patterns", "1.0.0", Active),
    ("tuple_indexing", "1.0.0", Accepted),
    ("associated_types", "1.0.0", Accepted),
    ("visible_private_types", "1.0.0", Active),
    ("slicing_syntax", "1.0.0", Active),
    ("box_syntax", "1.0.0", Active),
    ("on_unimplemented", "1.0.0", Active),
    ("simd_ffi", "1.0.0", Active),

    ("if_let", "1.0.0", Accepted),
    ("while_let", "1.0.0", Accepted),

    ("plugin", "1.0.0", Active),
    ("start", "1.0.0", Active),
    ("main", "1.0.0", Active),

    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    ("issue_5723_bootstrap", "1.0.0", Accepted),

    // A way to temporarily opt out of opt in copy. This will *never* be accepted.
    ("opt_out_copy", "1.0.0", Removed),

    // A way to temporarily opt out of the new orphan rules. This will *never* be accepted.
    ("old_orphan_check", "1.0.0", Deprecated),

    // A way to temporarily opt out of the new impl rules. This will *never* be accepted.
    ("old_impl_check", "1.0.0", Deprecated),

    // OIBIT specific features
    ("optin_builtin_traits", "1.0.0", Active),

    // int and uint are now deprecated
    ("int_uint", "1.0.0", Active),

    // macro reexport needs more discussion and stabilization
    ("macro_reexport", "1.0.0", Active),

    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    ("test_accepted_feature", "1.0.0", Accepted),
    ("test_removed_feature", "1.0.0", Removed),

    // Allows use of #[staged_api]
    ("staged_api", "1.0.0", Active),

    // Allows using items which are missing stability attributes
    ("unmarked_api", "1.0.0", Active)
];

enum Status {
    /// Represents an active feature that is currently being implemented or
    /// currently being considered for addition/removal.
    Active,

    /// Represents a feature gate that is temporarily enabling deprecated behavior.
    /// This gate will never be accepted.
    Deprecated,

    /// Represents a feature which has since been removed (it was once Active)
    Removed,

    /// This language feature has since been Accepted (it was once Active)
    Accepted,
}

/// A set of features to be used by later passes.
pub struct Features {
    pub unboxed_closures: bool,
    pub rustc_diagnostic_macros: bool,
    pub visible_private_types: bool,
    pub quote: bool,
    pub old_orphan_check: bool,
    pub simd_ffi: bool,
    pub unmarked_api: bool,
    /// spans of #![feature] attrs for stable language features. for error reporting
    pub declared_stable_lang_features: Vec<Span>,
    /// #![feature] attrs for non-language (library) features
    pub declared_lib_features: Vec<(InternedString, Span)>
}

impl Features {
    pub fn new() -> Features {
        Features {
            unboxed_closures: false,
            rustc_diagnostic_macros: false,
            visible_private_types: false,
            quote: false,
            old_orphan_check: false,
            simd_ffi: false,
            unmarked_api: false,
            declared_stable_lang_features: Vec::new(),
            declared_lib_features: Vec::new()
        }
    }
}

struct Context<'a> {
    features: Vec<&'static str>,
    span_handler: &'a SpanHandler,
    cm: &'a CodeMap,
}

impl<'a> Context<'a> {
    fn gate_feature(&self, feature: &str, span: Span, explain: &str) {
        if !self.has_feature(feature) {
            emit_feature_err(self.span_handler, feature, span, explain);
        }
    }

    fn warn_feature(&self, feature: &str, span: Span, explain: &str) {
        if !self.has_feature(feature) {
            emit_feature_warn(self.span_handler, feature, span, explain);
        }
    }
    fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|&n| n == feature)
    }
}

pub fn emit_feature_err(diag: &SpanHandler, feature: &str, span: Span, explain: &str) {
    diag.span_err(span, explain);
    diag.span_help(span, &format!("add #![feature({})] to the \
                                   crate attributes to enable",
                                  feature)[]);
}

pub fn emit_feature_warn(diag: &SpanHandler, feature: &str, span: Span, explain: &str) {
    diag.span_warn(span, explain);
    if diag.handler.can_emit_warnings {
        diag.span_help(span, &format!("add #![feature({})] to the \
                                       crate attributes to silence this warning",
                                      feature)[]);
    }
}

struct MacroVisitor<'a> {
    context: &'a Context<'a>
}

impl<'a, 'v> Visitor<'v> for MacroVisitor<'a> {
    fn visit_mac(&mut self, mac: &ast::Mac) {
        let ast::MacInvocTT(ref path, _, _) = mac.node;
        let id = path.segments.last().unwrap().identifier;

        if id == token::str_to_ident("asm") {
            self.context.gate_feature("asm", path.span, "inline assembly is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("log_syntax") {
            self.context.gate_feature("log_syntax", path.span, "`log_syntax!` is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("trace_macros") {
            self.context.gate_feature("trace_macros", path.span, "`trace_macros` is not \
                stable enough for use and is subject to change");
        }

        else if id == token::str_to_ident("concat_idents") {
            self.context.gate_feature("concat_idents", path.span, "`concat_idents` is not \
                stable enough for use and is subject to change");
        }
    }
}

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>
}

impl<'a> PostExpansionVisitor<'a> {
    fn gate_feature(&self, feature: &str, span: Span, explain: &str) {
        if !self.context.cm.span_is_internal(span) {
            self.context.gate_feature(feature, span, explain)
        }
    }
}

impl<'a, 'v> Visitor<'v> for PostExpansionVisitor<'a> {
    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !token::get_name(name).get().is_ascii() {
            self.gate_feature("non_ascii_idents", sp,
                              "non-ascii idents are not fully supported.");
        }
    }

    fn visit_item(&mut self, i: &ast::Item) {
        for attr in &i.attrs {
            if attr.name() == "thread_local" {
                self.gate_feature("thread_local", i.span,
                                  "`#[thread_local]` is an experimental feature, and does not \
                                  currently handle destructors. There is no corresponding \
                                  `#[task_local]` mapping to the task model");
            } else if attr.name() == "linkage" {
                self.gate_feature("linkage", i.span,
                                  "the `linkage` attribute is experimental \
                                   and not portable across platforms")
            } else if attr.name() == "rustc_on_unimplemented" {
                self.gate_feature("on_unimplemented", i.span,
                                  "the `#[rustc_on_unimplemented]` attribute \
                                  is an experimental feature")
            }
        }
        match i.node {
            ast::ItemExternCrate(_) => {
                if attr::contains_name(&i.attrs[], "plugin") {
                    self.gate_feature("plugin", i.span,
                                      "compiler plugins are experimental \
                                       and possibly buggy");
                } else if attr::contains_name(&i.attrs[], "macro_reexport") {
                    self.gate_feature("macro_reexport", i.span,
                                      "macros reexports are experimental \
                                       and possibly buggy");
                }
            }

            ast::ItemForeignMod(ref foreign_module) => {
                if attr::contains_name(&i.attrs[], "link_args") {
                    self.gate_feature("link_args", i.span,
                                      "the `link_args` attribute is not portable \
                                       across platforms, it is recommended to \
                                       use `#[link(name = \"foo\")]` instead")
                }
                if foreign_module.abi == RustIntrinsic {
                    self.gate_feature("intrinsics",
                                      i.span,
                                      "intrinsics are subject to change")
                }
            }

            ast::ItemFn(..) => {
                if attr::contains_name(&i.attrs[], "plugin_registrar") {
                    self.gate_feature("plugin_registrar", i.span,
                                      "compiler plugins are experimental and possibly buggy");
                }
                if attr::contains_name(&i.attrs[], "start") {
                    self.gate_feature("start", i.span,
                                      "a #[start] function is an experimental \
                                       feature whose signature may change \
                                       over time");
                }
                if attr::contains_name(&i.attrs[], "main") {
                    self.gate_feature("main", i.span,
                                      "declaration of a nonstandard #[main] \
                                       function may change over time, for now \
                                       a top-level `fn main()` is required");
                }
            }

            ast::ItemStruct(..) => {
                if attr::contains_name(&i.attrs[], "simd") {
                    self.gate_feature("simd", i.span,
                                      "SIMD types are experimental and possibly buggy");
                }
            }

            ast::ItemImpl(_, polarity, _, _, _, _) => {
                match polarity {
                    ast::ImplPolarity::Negative => {
                        self.gate_feature("optin_builtin_traits",
                                          i.span,
                                          "negative trait bounds are not yet fully implemented; \
                                          use marker types for now");
                    },
                    _ => {}
                }

                if attr::contains_name(&i.attrs,
                                       "unsafe_destructor") {
                    self.gate_feature("unsafe_destructor",
                                      i.span,
                                      "`#[unsafe_destructor]` allows too \
                                       many unsafe patterns and may be \
                                       removed in the future");
                }

                if attr::contains_name(&i.attrs[],
                                       "old_orphan_check") {
                    self.gate_feature(
                        "old_orphan_check",
                        i.span,
                        "the new orphan check rules will eventually be strictly enforced");
                }

                if attr::contains_name(&i.attrs[],
                                       "old_impl_check") {
                    self.gate_feature("old_impl_check",
                                      i.span,
                                      "`#[old_impl_check]` will be removed in the future");
                }
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        if attr::contains_name(&i.attrs[], "linkage") {
            self.gate_feature("linkage", i.span,
                              "the `linkage` attribute is experimental \
                               and not portable across platforms")
        }

        let links_to_llvm = match attr::first_attr_value_str_by_name(&i.attrs,
                                                                     "link_name") {
            Some(val) => val.get().starts_with("llvm."),
            _ => false
        };
        if links_to_llvm {
            self.gate_feature("link_llvm_intrinsics", i.span,
                              "linking to LLVM intrinsics is experimental");
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        match t.node {
            ast::TyPath(ref p, _) => {
                match &*p.segments {

                    [ast::PathSegment { identifier, .. }] => {
                        let name = token::get_ident(identifier);
                        let msg = if name == "int" {
                            Some("the `int` type is deprecated; \
                                  use `isize` or a fixed-sized integer")
                        } else if name == "uint" {
                            Some("the `uint` type is deprecated; \
                                  use `usize` or a fixed-sized integer")
                        } else {
                            None
                        };

                        if let Some(msg) = msg {
                            self.context.warn_feature("int_uint", t.span, msg)
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        visit::walk_ty(self, t);
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        match e.node {
            ast::ExprBox(..) | ast::ExprUnary(ast::UnOp::UnUniq, _) => {
                self.gate_feature("box_syntax",
                                  e.span,
                                  "box expression syntax is experimental in alpha release; \
                                   you can call `Box::new` instead.");
            }
            ast::ExprLit(ref lit) => {
                match lit.node {
                    ast::LitInt(_, ty) => {
                        let msg = if let ast::SignedIntLit(ast::TyIs(true), _) = ty {
                            Some("the `i` suffix on integers is deprecated; use `is` \
                                  or one of the fixed-sized suffixes")
                        } else if let ast::UnsignedIntLit(ast::TyUs(true)) = ty {
                            Some("the `u` suffix on integers is deprecated; use `us` \
                                 or one of the fixed-sized suffixes")
                        } else {
                            None
                        };
                        if let Some(msg) = msg {
                            self.context.warn_feature("int_uint", e.span, msg);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if attr.check_name("staged_api") {
            self.gate_feature("staged_api", attr.span,
                              "staged_api is for use by rustc only");
        }

        if attr::contains_name(slice::ref_slice(attr), "lang") {
            self.gate_feature("lang_items",
                              attr.span,
                              "language items are subject to change");
        }
    }

    fn visit_pat(&mut self, pattern: &ast::Pat) {
        match pattern.node {
            ast::PatVec(_, Some(_), ref last) if !last.is_empty() => {
                self.gate_feature("advanced_slice_patterns",
                                  pattern.span,
                                  "multiple-element slice matches anywhere \
                                   but at the end of a slice (e.g. \
                                   `[0, ..xs, 0]` are experimental")
            }
            ast::PatBox(..) => {
                self.gate_feature("box_syntax",
                                  pattern.span,
                                  "box pattern syntax is experimental in alpha release");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(&mut self,
                fn_kind: visit::FnKind<'v>,
                fn_decl: &'v ast::FnDecl,
                block: &'v ast::Block,
                span: Span,
                _node_id: NodeId) {
        match fn_kind {
            visit::FkItemFn(_, _, _, abi) if abi == RustIntrinsic => {
                self.gate_feature("intrinsics",
                                  span,
                                  "intrinsics are subject to change")
            }
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, block, span);
    }
}

fn check_crate_inner<F>(cm: &CodeMap, span_handler: &SpanHandler, krate: &ast::Crate,
                        check: F)
                       -> Features
    where F: FnOnce(&mut Context, &ast::Crate)
{
    let mut cx = Context {
        features: Vec::new(),
        span_handler: span_handler,
        cm: cm,
    };

    let mut accepted_features = Vec::new();
    let mut unknown_features = Vec::new();

    for attr in &krate.attrs {
        if !attr.check_name("feature") {
            continue
        }

        match attr.meta_item_list() {
            None => {
                span_handler.span_err(attr.span, "malformed feature attribute, \
                                                  expected #![feature(...)]");
            }
            Some(list) => {
                for mi in list {
                    let name = match mi.node {
                        ast::MetaWord(ref word) => (*word).clone(),
                        _ => {
                            span_handler.span_err(mi.span,
                                                  "malformed feature, expected just \
                                                   one word");
                            continue
                        }
                    };
                    match KNOWN_FEATURES.iter()
                                        .find(|& &(n, _, _)| name == n) {
                        Some(&(name, _, Active)) => {
                            cx.features.push(name);
                        }
                        Some(&(name, _, Deprecated)) => {
                            cx.features.push(name);
                            span_handler.span_warn(
                                mi.span,
                                "feature is deprecated and will only be available \
                                 for a limited time, please rewrite code that relies on it");
                        }
                        Some(&(_, _, Removed)) => {
                            span_handler.span_err(mi.span, "feature has been removed");
                        }
                        Some(&(_, _, Accepted)) => {
                            accepted_features.push(mi.span);
                        }
                        None => {
                            unknown_features.push((name, mi.span));
                        }
                    }
                }
            }
        }
    }

    check(&mut cx, krate);

    Features {
        unboxed_closures: cx.has_feature("unboxed_closures"),
        rustc_diagnostic_macros: cx.has_feature("rustc_diagnostic_macros"),
        visible_private_types: cx.has_feature("visible_private_types"),
        quote: cx.has_feature("quote"),
        old_orphan_check: cx.has_feature("old_orphan_check"),
        simd_ffi: cx.has_feature("simd_ffi"),
        unmarked_api: cx.has_feature("unmarked_api"),
        declared_stable_lang_features: accepted_features,
        declared_lib_features: unknown_features
    }
}

pub fn check_crate_macros(cm: &CodeMap, span_handler: &SpanHandler, krate: &ast::Crate)
-> Features {
    check_crate_inner(cm, span_handler, krate,
                      |ctx, krate| visit::walk_crate(&mut MacroVisitor { context: ctx }, krate))
}

pub fn check_crate(cm: &CodeMap, span_handler: &SpanHandler, krate: &ast::Crate)
-> Features {
    check_crate_inner(cm, span_handler, krate,
                      |ctx, krate| visit::walk_crate(&mut PostExpansionVisitor { context: ctx },
                                                     krate))
}

