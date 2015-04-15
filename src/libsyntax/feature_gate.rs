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
use self::AttributeType::*;

use abi::Abi;
use ast::NodeId;
use ast;
use attr;
use attr::AttrMetaMethods;
use codemap::{CodeMap, Span};
use diagnostic::SpanHandler;
use visit;
use visit::Visitor;
use parse::token::{self, InternedString};

use std::ascii::AsciiExt;

// If you change this list without updating src/doc/reference.md, @cmr will be sad
// Don't ever remove anything from this list; set them to 'Removed'.
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
// NB: The featureck.py script parses this information directly out of the source
// so take care when modifying it.
const KNOWN_FEATURES: &'static [(&'static str, &'static str, Status)] = &[
    ("globs", "1.0.0", Accepted),
    ("macro_rules", "1.0.0", Accepted),
    ("struct_variant", "1.0.0", Accepted),
    ("asm", "1.0.0", Active),
    ("managed_boxes", "1.0.0", Removed),
    ("non_ascii_idents", "1.0.0", Active),
    ("thread_local", "1.0.0", Active),
    ("link_args", "1.0.0", Active),
    ("plugin_registrar", "1.0.0", Active),
    ("log_syntax", "1.0.0", Active),
    ("trace_macros", "1.0.0", Active),
    ("concat_idents", "1.0.0", Active),
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
    ("reflect", "1.0.0", Active),
    ("import_shadowing", "1.0.0", Removed),
    ("advanced_slice_patterns", "1.0.0", Active),
    ("tuple_indexing", "1.0.0", Accepted),
    ("associated_types", "1.0.0", Accepted),
    ("visible_private_types", "1.0.0", Active),
    ("slicing_syntax", "1.0.0", Accepted),
    ("box_syntax", "1.0.0", Active),
    ("on_unimplemented", "1.0.0", Active),
    ("simd_ffi", "1.0.0", Active),
    ("allocator", "1.0.0", Active),

    ("if_let", "1.0.0", Accepted),
    ("while_let", "1.0.0", Accepted),

    ("plugin", "1.0.0", Active),
    ("start", "1.0.0", Active),
    ("main", "1.0.0", Active),

    ("fundamental", "1.0.0", Active),

    // Deprecate after snapshot
    // SNAP 5520801
    ("unsafe_destructor", "1.0.0", Active),

    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    ("issue_5723_bootstrap", "1.0.0", Accepted),

    // A way to temporarily opt out of opt in copy. This will *never* be accepted.
    ("opt_out_copy", "1.0.0", Removed),

    // OIBIT specific features
    ("optin_builtin_traits", "1.0.0", Active),

    // macro reexport needs more discussion and stabilization
    ("macro_reexport", "1.0.0", Active),

    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    ("test_accepted_feature", "1.0.0", Accepted),
    ("test_removed_feature", "1.0.0", Removed),

    // Allows use of #[staged_api]
    ("staged_api", "1.0.0", Active),

    // Allows using items which are missing stability attributes
    ("unmarked_api", "1.0.0", Active),

    // Allows using #![no_std]
    ("no_std", "1.0.0", Active),

    // Allows using `box` in patterns; RFC 469
    ("box_patterns", "1.0.0", Active),

    // Allows using the unsafe_no_drop_flag attribute (unlikely to
    // switch to Accepted; see RFC 320)
    ("unsafe_no_drop_flag", "1.0.0", Active),

    // Allows the use of custom attributes; RFC 572
    ("custom_attribute", "1.0.0", Active),

    // Allows the use of #[derive(Anything)] as sugar for
    // #[derive_Anything].
    ("custom_derive", "1.0.0", Active),

    // Allows the use of rustc_* attributes; RFC 572
    ("rustc_attrs", "1.0.0", Active),

    // Allows the use of `static_assert`
    ("static_assert", "1.0.0", Active),

    // Allows the use of #[allow_internal_unstable]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    ("allow_internal_unstable", "1.0.0", Active),

    // #23121. Array patterns have some hazards yet.
    ("slice_patterns", "1.0.0", Active),

    // Allows use of unary negate on unsigned integers, e.g. -e for e: u8
    ("negate_unsigned", "1.0.0", Active),
];
// (changing above list without updating src/doc/reference.md makes @cmr sad)

enum Status {
    /// Represents an active feature that is currently being implemented or
    /// currently being considered for addition/removal.
    Active,

    /// Represents a feature which has since been removed (it was once Active)
    Removed,

    /// This language feature has since been Accepted (it was once Active)
    Accepted,
}

// Attributes that have a special meaning to rustc or rustdoc
pub const KNOWN_ATTRIBUTES: &'static [(&'static str, AttributeType)] = &[
    // Normal attributes

    ("warn", Normal),
    ("allow", Normal),
    ("forbid", Normal),
    ("deny", Normal),

    ("macro_reexport", Normal),
    ("macro_use", Normal),
    ("macro_export", Normal),
    ("plugin_registrar", Normal),

    ("cfg", Normal),
    ("cfg_attr", Normal),
    ("main", Normal),
    ("start", Normal),
    ("test", Normal),
    ("bench", Normal),
    ("simd", Normal),
    ("repr", Normal),
    ("path", Normal),
    ("abi", Normal),
    ("automatically_derived", Normal),
    ("no_mangle", Normal),
    ("no_link", Normal),
    ("derive", Normal),
    ("should_panic", Normal),
    ("ignore", Normal),
    ("no_implicit_prelude", Normal),
    ("reexport_test_harness_main", Normal),
    ("link_args", Normal),
    ("macro_escape", Normal),

    ("unsafe_destructor", Gated("unsafe_destructor",
                                "`#[unsafe_destructor]` does nothing anymore")),
    ("staged_api", Gated("staged_api",
                         "staged_api is for use by rustc only")),
    ("plugin", Gated("plugin",
                     "compiler plugins are experimental \
                      and possibly buggy")),
    ("no_std", Gated("no_std",
                     "no_std is experimental")),
    ("lang", Gated("lang_items",
                     "language items are subject to change")),
    ("linkage", Gated("linkage",
                      "the `linkage` attribute is experimental \
                       and not portable across platforms")),
    ("thread_local", Gated("thread_local",
                            "`#[thread_local]` is an experimental feature, and does not \
                             currently handle destructors. There is no corresponding \
                             `#[task_local]` mapping to the task model")),

    ("rustc_on_unimplemented", Gated("on_unimplemented",
                                     "the `#[rustc_on_unimplemented]` attribute \
                                      is an experimental feature")),
    ("allocator", Gated("allocator",
                        "the `#[allocator]` attribute is an experimental feature")),
    ("rustc_variance", Gated("rustc_attrs",
                             "the `#[rustc_variance]` attribute \
                              is an experimental feature")),
    ("rustc_error", Gated("rustc_attrs",
                          "the `#[rustc_error]` attribute \
                           is an experimental feature")),
    ("rustc_move_fragments", Gated("rustc_attrs",
                                   "the `#[rustc_move_fragments]` attribute \
                                    is an experimental feature")),

    ("allow_internal_unstable", Gated("allow_internal_unstable",
                                      EXPLAIN_ALLOW_INTERNAL_UNSTABLE)),

    ("fundamental", Gated("fundamental",
                          "the `#[fundamental]` attribute \
                           is an experimental feature")),

    // FIXME: #14408 whitelist docs since rustdoc looks at them
    ("doc", Whitelisted),

    // FIXME: #14406 these are processed in trans, which happens after the
    // lint pass
    ("cold", Whitelisted),
    ("export_name", Whitelisted),
    ("inline", Whitelisted),
    ("link", Whitelisted),
    ("link_name", Whitelisted),
    ("link_section", Whitelisted),
    ("no_builtins", Whitelisted),
    ("no_mangle", Whitelisted),
    ("no_stack_check", Whitelisted),
    ("packed", Whitelisted),
    ("static_assert", Gated("static_assert",
                            "`#[static_assert]` is an experimental feature, and has a poor API")),
    ("no_debug", Whitelisted),
    ("omit_gdb_pretty_printer_section", Whitelisted),
    ("unsafe_no_drop_flag", Gated("unsafe_no_drop_flag",
                                  "unsafe_no_drop_flag has unstable semantics \
                                   and may be removed in the future")),

    // used in resolve
    ("prelude_import", Whitelisted),

    // FIXME: #14407 these are only looked at on-demand so we can't
    // guarantee they'll have already been checked
    ("deprecated", Whitelisted),
    ("must_use", Whitelisted),
    ("stable", Whitelisted),
    ("unstable", Whitelisted),

    ("rustc_paren_sugar", Gated("unboxed_closures",
                                "unboxed_closures are still evolving")),
    ("rustc_reflect_like", Gated("reflect",
                                 "defining reflective traits is still evolving")),

    // Crate level attributes
    ("crate_name", CrateLevel),
    ("crate_type", CrateLevel),
    ("crate_id", CrateLevel),
    ("feature", CrateLevel),
    ("no_start", CrateLevel),
    ("no_main", CrateLevel),
    ("no_builtins", CrateLevel),
    ("recursion_limit", CrateLevel),
];

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum AttributeType {
    /// Normal, builtin attribute that is consumed
    /// by the compiler before the unused_attribute check
    Normal,

    /// Builtin attribute that may not be consumed by the compiler
    /// before the unused_attribute check. These attributes
    /// will be ignored by the unused_attribute lint
    Whitelisted,

    /// Is gated by a given feature gate and reason
    /// These get whitelisted too
    Gated(&'static str, &'static str),

    /// Builtin attribute that is only allowed at the crate level
    CrateLevel,
}

/// A set of features to be used by later passes.
pub struct Features {
    pub unboxed_closures: bool,
    pub rustc_diagnostic_macros: bool,
    pub visible_private_types: bool,
    pub allow_quote: bool,
    pub allow_asm: bool,
    pub allow_log_syntax: bool,
    pub allow_concat_idents: bool,
    pub allow_trace_macros: bool,
    pub allow_internal_unstable: bool,
    pub allow_custom_derive: bool,
    pub simd_ffi: bool,
    pub unmarked_api: bool,
    pub negate_unsigned: bool,
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
            allow_quote: false,
            allow_asm: false,
            allow_log_syntax: false,
            allow_concat_idents: false,
            allow_trace_macros: false,
            allow_internal_unstable: false,
            allow_custom_derive: false,
            simd_ffi: false,
            unmarked_api: false,
            negate_unsigned: false,
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
        let has_feature = self.has_feature(feature);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", feature, span, has_feature);
        if !has_feature {
            emit_feature_err(self.span_handler, feature, span, explain);
        }
    }
    fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|&n| n == feature)
    }

    fn check_attribute(&self, attr: &ast::Attribute) {
        debug!("check_attribute(attr = {:?})", attr);
        let name = &*attr.name();
        for &(n, ty) in KNOWN_ATTRIBUTES {
            if n == name {
                if let Gated(gate, desc) = ty {
                    self.gate_feature(gate, attr.span, desc);
                }
                debug!("check_attribute: {:?} is known, {:?}", name, ty);
                return;
            }
        }
        if name.starts_with("rustc_") {
            self.gate_feature("rustc_attrs", attr.span,
                              "unless otherwise specified, attributes \
                               with the prefix `rustc_` \
                               are reserved for internal compiler diagnostics");
        } else if name.starts_with("derive_") {
            self.gate_feature("custom_derive", attr.span,
                              "attributes of the form `#[derive_*]` are reserved
                               for the compiler");
        } else {
            self.gate_feature("custom_attribute", attr.span,
                       &format!("The attribute `{}` is currently \
                                unknown to the compiler and \
                                may have meaning \
                                added to it in the future",
                                name));
        }
    }
}

pub fn emit_feature_err(diag: &SpanHandler, feature: &str, span: Span, explain: &str) {
    diag.span_err(span, explain);

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some() { return; }
    diag.fileline_help(span, &format!("add #![feature({})] to the \
                                   crate attributes to enable",
                                  feature));
}

pub fn emit_feature_warn(diag: &SpanHandler, feature: &str, span: Span, explain: &str) {
    diag.span_warn(span, explain);

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some() { return; }
    if diag.handler.can_emit_warnings {
        diag.fileline_help(span, &format!("add #![feature({})] to the \
                                       crate attributes to silence this warning",
                                      feature));
    }
}

pub const EXPLAIN_ASM: &'static str =
    "inline assembly is not stable enough for use and is subject to change";

pub const EXPLAIN_LOG_SYNTAX: &'static str =
    "`log_syntax!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CONCAT_IDENTS: &'static str =
    "`concat_idents` is not stable enough for use and is subject to change";

pub const EXPLAIN_TRACE_MACROS: &'static str =
    "`trace_macros` is not stable enough for use and is subject to change";
pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &'static str =
    "allow_internal_unstable side-steps feature gating and stability checks";

pub const EXPLAIN_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is not stable enough for use and is subject to change";

struct MacroVisitor<'a> {
    context: &'a Context<'a>
}

impl<'a, 'v> Visitor<'v> for MacroVisitor<'a> {
    fn visit_mac(&mut self, mac: &ast::Mac) {
        let ast::MacInvocTT(ref path, _, _) = mac.node;
        let id = path.segments.last().unwrap().identifier;

        // Issue 22234: If you add a new case here, make sure to also
        // add code to catch the macro during or after expansion.
        //
        // We still keep this MacroVisitor (rather than *solely*
        // relying on catching cases during or after expansion) to
        // catch uses of these macros within conditionally-compiled
        // code, e.g. `#[cfg]`-guarded functions.

        if id == token::str_to_ident("asm") {
            self.context.gate_feature("asm", path.span, EXPLAIN_ASM);
        }

        else if id == token::str_to_ident("log_syntax") {
            self.context.gate_feature("log_syntax", path.span, EXPLAIN_LOG_SYNTAX);
        }

        else if id == token::str_to_ident("trace_macros") {
            self.context.gate_feature("trace_macros", path.span, EXPLAIN_TRACE_MACROS);
        }

        else if id == token::str_to_ident("concat_idents") {
            self.context.gate_feature("concat_idents", path.span, EXPLAIN_CONCAT_IDENTS);
        }
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        self.context.check_attribute(attr);
    }
}

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>
}

impl<'a> PostExpansionVisitor<'a> {
    fn gate_feature(&self, feature: &str, span: Span, explain: &str) {
        if !self.context.cm.span_allows_unstable(span) {
            self.context.gate_feature(feature, span, explain)
        }
    }
}

impl<'a, 'v> Visitor<'v> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if !self.context.cm.span_allows_unstable(attr.span) {
            self.context.check_attribute(attr);
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !token::get_name(name).is_ascii() {
            self.gate_feature("non_ascii_idents", sp,
                              "non-ascii idents are not fully supported.");
        }
    }

    fn visit_item(&mut self, i: &ast::Item) {
        match i.node {
            ast::ItemExternCrate(_) => {
                if attr::contains_name(&i.attrs[..], "macro_reexport") {
                    self.gate_feature("macro_reexport", i.span,
                                      "macros reexports are experimental \
                                       and possibly buggy");
                }
            }

            ast::ItemForeignMod(ref foreign_module) => {
                if attr::contains_name(&i.attrs[..], "link_args") {
                    self.gate_feature("link_args", i.span,
                                      "the `link_args` attribute is not portable \
                                       across platforms, it is recommended to \
                                       use `#[link(name = \"foo\")]` instead")
                }
                if foreign_module.abi == Abi::RustIntrinsic {
                    self.gate_feature("intrinsics",
                                      i.span,
                                      "intrinsics are subject to change")
                }
            }

            ast::ItemFn(..) => {
                if attr::contains_name(&i.attrs[..], "plugin_registrar") {
                    self.gate_feature("plugin_registrar", i.span,
                                      "compiler plugins are experimental and possibly buggy");
                }
                if attr::contains_name(&i.attrs[..], "start") {
                    self.gate_feature("start", i.span,
                                      "a #[start] function is an experimental \
                                       feature whose signature may change \
                                       over time");
                }
                if attr::contains_name(&i.attrs[..], "main") {
                    self.gate_feature("main", i.span,
                                      "declaration of a nonstandard #[main] \
                                       function may change over time, for now \
                                       a top-level `fn main()` is required");
                }
            }

            ast::ItemStruct(..) => {
                if attr::contains_name(&i.attrs[..], "simd") {
                    self.gate_feature("simd", i.span,
                                      "SIMD types are experimental and possibly buggy");
                }
            }

            ast::ItemDefaultImpl(..) => {
                self.gate_feature("optin_builtin_traits",
                                  i.span,
                                  "default trait implementations are experimental \
                                   and possibly buggy");
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
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        let links_to_llvm = match attr::first_attr_value_str_by_name(&i.attrs,
                                                                     "link_name") {
            Some(val) => val.starts_with("llvm."),
            _ => false
        };
        if links_to_llvm {
            self.gate_feature("link_llvm_intrinsics", i.span,
                              "linking to LLVM intrinsics is experimental");
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        match e.node {
            ast::ExprBox(..) | ast::ExprUnary(ast::UnOp::UnUniq, _) => {
                self.gate_feature("box_syntax",
                                  e.span,
                                  "box expression syntax is experimental; \
                                   you can call `Box::new` instead.");
            }
            _ => {}
        }
        visit::walk_expr(self, e);
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
            ast::PatVec(..) => {
                self.gate_feature("slice_patterns",
                                  pattern.span,
                                  "slice pattern syntax is experimental");
            }
            ast::PatBox(..) => {
                self.gate_feature("box_patterns",
                                  pattern.span,
                                  "box pattern syntax is experimental");
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
            visit::FkItemFn(_, _, _, abi, _) if abi == Abi::RustIntrinsic => {
                self.gate_feature("intrinsics",
                                  span,
                                  "intrinsics are subject to change")
            }
            visit::FkItemFn(_, _, _, abi, _) |
            visit::FkMethod(_, &ast::MethodSig { abi, .. }, _) if abi == Abi::RustCall => {
                self.gate_feature("unboxed_closures",
                                  span,
                                  "rust-call ABI is subject to change")
            }
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, block, span);
    }
}

fn check_crate_inner<F>(cm: &CodeMap, span_handler: &SpanHandler,
                        krate: &ast::Crate,
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

    // FIXME (pnkfelix): Before adding the 99th entry below, change it
    // to a single-pass (instead of N calls to `.has_feature`).

    Features {
        unboxed_closures: cx.has_feature("unboxed_closures"),
        rustc_diagnostic_macros: cx.has_feature("rustc_diagnostic_macros"),
        visible_private_types: cx.has_feature("visible_private_types"),
        allow_quote: cx.has_feature("quote"),
        allow_asm: cx.has_feature("asm"),
        allow_log_syntax: cx.has_feature("log_syntax"),
        allow_concat_idents: cx.has_feature("concat_idents"),
        allow_trace_macros: cx.has_feature("trace_macros"),
        allow_internal_unstable: cx.has_feature("allow_internal_unstable"),
        allow_custom_derive: cx.has_feature("custom_derive"),
        simd_ffi: cx.has_feature("simd_ffi"),
        unmarked_api: cx.has_feature("unmarked_api"),
        negate_unsigned: cx.has_feature("negate_unsigned"),
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
                   -> Features
{
    check_crate_inner(cm, span_handler, krate,
                      |ctx, krate| visit::walk_crate(&mut PostExpansionVisitor { context: ctx },
                                                     krate))
}
