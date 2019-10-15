use super::{active::{ACTIVE_FEATURES, Features}, Feature, State as FeatureState};
use super::accepted::ACCEPTED_FEATURES;
use super::removed::{REMOVED_FEATURES, STABLE_REMOVED_FEATURES};
use super::builtin_attrs::{AttributeGate, BUILTIN_ATTRIBUTE_MAP};

use crate::ast::{
    self, AssocTyConstraint, AssocTyConstraintKind, NodeId, GenericParam, GenericParamKind,
    PatKind, RangeEnd, VariantData,
};
use crate::attr::{self, check_builtin_attribute};
use crate::source_map::Spanned;
use crate::edition::{ALL_EDITIONS, Edition};
use crate::visit::{self, FnKind, Visitor};
use crate::parse::token;
use crate::sess::ParseSess;
use crate::symbol::{Symbol, sym};
use crate::tokenstream::TokenTree;

use errors::{Applicability, DiagnosticBuilder, Handler};
use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::abi::Abi;
use syntax_pos::{Span, DUMMY_SP, MultiSpan};
use log::debug;

use std::env;

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // First argument is tracking issue link; second argument is an optional
    // help message, which defaults to "remove this attribute"
    Deprecated(&'static str, Option<&'static str>),
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $level: expr) => {{
        let (cx, has_feature, span,
             name, explain, level) = (&*$cx, $has_feature, $span, $name, $explain, $level);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            leveled_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain, level)
                .emit();
        }
    }}
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         sym::$feature, $explain, GateStrength::Hard)
    };
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         sym::$feature, $explain, $level)
    };
}

crate fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess, features: &Features) {
    PostExpansionVisitor { parse_sess, features }.visit_attribute(attr)
}

fn find_lang_feature_issue(feature: Symbol) -> Option<u32> {
    if let Some(info) = ACTIVE_FEATURES.iter().find(|t| t.name == feature) {
        // FIXME (#28244): enforce that active features have issue numbers
        // assert!(info.issue.is_some())
        info.issue
    } else {
        // search in Accepted, Removed, or Stable Removed features
        let found = ACCEPTED_FEATURES.iter().chain(REMOVED_FEATURES).chain(STABLE_REMOVED_FEATURES)
            .find(|t| t.name == feature);
        match found {
            Some(&Feature { issue, .. }) => issue,
            None => panic!("Feature `{}` is not declared anywhere", feature),
        }
    }
}

pub enum GateIssue {
    Language,
    Library(Option<u32>)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum GateStrength {
    /// A hard error. (Most feature gates should use this.)
    Hard,
    /// Only a warning. (Use this only as backwards-compatibility demands.)
    Soft,
}

pub fn emit_feature_err(
    sess: &ParseSess,
    feature: Symbol,
    span: Span,
    issue: GateIssue,
    explain: &str,
) {
    feature_err(sess, feature, span, issue, explain).emit();
}

pub fn feature_err<'a, S: Into<MultiSpan>>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: S,
    issue: GateIssue,
    explain: &str,
) -> DiagnosticBuilder<'a> {
    leveled_feature_err(sess, feature, span, issue, explain, GateStrength::Hard)
}

fn leveled_feature_err<'a, S: Into<MultiSpan>>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: S,
    issue: GateIssue,
    explain: &str,
    level: GateStrength,
) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

    let issue = match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    };

    let mut err = match level {
        GateStrength::Hard => {
            diag.struct_span_err_with_code(span, explain, stringify_error_code!(E0658))
        }
        GateStrength::Soft => diag.struct_span_warn(span, explain),
    };

    match issue {
        None | Some(0) => {}  // We still accept `0` as a stand-in for backwards compatibility
        Some(n) => {
            err.note(&format!(
                "for more information, see https://github.com/rust-lang/rust/issues/{}",
                n,
            ));
        }
    }

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add `#![feature({})]` to the crate attributes to enable", feature));
    }

    // If we're on stable and only emitting a "soft" warning, add a note to
    // clarify that the feature isn't "on" (rather than being on but
    // warning-worthy).
    if !sess.unstable_features.is_nightly_build() && level == GateStrength::Soft {
        err.help("a nightly build of the compiler is required to enable this feature");
    }

    err

}

const EXPLAIN_BOX_SYNTAX: &str =
    "box expression syntax is experimental; you can call `Box::new` instead";

pub const EXPLAIN_STMT_ATTR_SYNTAX: &str =
    "attributes on expressions are experimental";

pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &str =
    "allow_internal_unstable side-steps feature gating and stability checks";
pub const EXPLAIN_ALLOW_INTERNAL_UNSAFE: &str =
    "allow_internal_unsafe side-steps the unsafe_code lint";

pub const EXPLAIN_UNSIZED_TUPLE_COERCION: &str =
    "unsized tuple coercion is not stable enough for use and is subject to change";

struct PostExpansionVisitor<'a> {
    parse_sess: &'a ParseSess,
    features: &'a Features,
}

macro_rules! gate_feature_post {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable(sym::$feature) {
            gate_feature!(cx, $feature, span, $explain)
        }
    }};
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable(sym::$feature) {
            gate_feature!(cx, $feature, span, $explain, $level)
        }
    }}
}

impl<'a> PostExpansionVisitor<'a> {
    fn check_abi(&self, abi: Abi, span: Span) {
        match abi {
            Abi::RustIntrinsic => {
                gate_feature_post!(&self, intrinsics, span,
                                   "intrinsics are subject to change");
            },
            Abi::PlatformIntrinsic => {
                gate_feature_post!(&self, platform_intrinsics, span,
                                   "platform intrinsics are experimental and possibly buggy");
            },
            Abi::Vectorcall => {
                gate_feature_post!(&self, abi_vectorcall, span,
                                   "vectorcall is experimental and subject to change");
            },
            Abi::Thiscall => {
                gate_feature_post!(&self, abi_thiscall, span,
                                   "thiscall is experimental and subject to change");
            },
            Abi::RustCall => {
                gate_feature_post!(&self, unboxed_closures, span,
                                   "rust-call ABI is subject to change");
            },
            Abi::PtxKernel => {
                gate_feature_post!(&self, abi_ptx, span,
                                   "PTX ABIs are experimental and subject to change");
            },
            Abi::Unadjusted => {
                gate_feature_post!(&self, abi_unadjusted, span,
                                   "unadjusted ABI is an implementation detail and perma-unstable");
            },
            Abi::Msp430Interrupt => {
                gate_feature_post!(&self, abi_msp430_interrupt, span,
                                   "msp430-interrupt ABI is experimental and subject to change");
            },
            Abi::X86Interrupt => {
                gate_feature_post!(&self, abi_x86_interrupt, span,
                                   "x86-interrupt ABI is experimental and subject to change");
            },
            Abi::AmdGpuKernel => {
                gate_feature_post!(&self, abi_amdgpu_kernel, span,
                                   "amdgpu-kernel ABI is experimental and subject to change");
            },
            // Stable
            Abi::Cdecl |
            Abi::Stdcall |
            Abi::Fastcall |
            Abi::Aapcs |
            Abi::Win64 |
            Abi::SysV64 |
            Abi::Rust |
            Abi::C |
            Abi::System => {}
        }
    }

    fn maybe_report_invalid_custom_discriminants(&self, variants: &[ast::Variant]) {
        let has_fields = variants.iter().any(|variant| match variant.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => true,
            VariantData::Unit(..) => false,
        });

        let discriminant_spans = variants.iter().filter(|variant| match variant.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => false,
            VariantData::Unit(..) => true,
        })
        .filter_map(|variant| variant.disr_expr.as_ref().map(|c| c.value.span))
        .collect::<Vec<_>>();

        if !discriminant_spans.is_empty() && has_fields {
            let mut err = feature_err(
                self.parse_sess,
                sym::arbitrary_enum_discriminant,
                discriminant_spans.clone(),
                crate::feature_gate::GateIssue::Language,
                "custom discriminant values are not allowed in enums with tuple or struct variants",
            );
            for sp in discriminant_spans {
                err.span_label(sp, "disallowed custom discriminant");
            }
            for variant in variants.iter() {
                match &variant.data {
                    VariantData::Struct(..) => {
                        err.span_label(
                            variant.span,
                            "struct variant defined here",
                        );
                    }
                    VariantData::Tuple(..) => {
                        err.span_label(
                            variant.span,
                            "tuple variant defined here",
                        );
                    }
                    VariantData::Unit(..) => {}
                }
            }
            err.emit();
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        let attr_info =
            attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name)).map(|a| **a);
        // Check feature gates for built-in attributes.
        if let Some((.., AttributeGate::Gated(_, name, descr, has_feature))) = attr_info {
            gate_feature_fn!(self, has_feature, attr.span, name, descr, GateStrength::Hard);
        }
        // Check input tokens for built-in and key-value attributes.
        match attr_info {
            // `rustc_dummy` doesn't have any restrictions specific to built-in attributes.
            Some((name, _, template, _)) if name != sym::rustc_dummy =>
                check_builtin_attribute(self.parse_sess, attr, name, template),
            _ => if let Some(TokenTree::Token(token)) = attr.tokens.trees().next() {
                if token == token::Eq {
                    // All key-value attributes are restricted to meta-item syntax.
                    attr.parse_meta(self.parse_sess).map_err(|mut err| err.emit()).ok();
                }
            }
        }
        // Check unstable flavors of the `#[doc]` attribute.
        if attr.check_name(sym::doc) {
            for nested_meta in attr.meta_item_list().unwrap_or_default() {
                macro_rules! gate_doc { ($($name:ident => $feature:ident)*) => {
                    $(if nested_meta.check_name(sym::$name) {
                        let msg = concat!("`#[doc(", stringify!($name), ")]` is experimental");
                        gate_feature!(self, $feature, attr.span, msg);
                    })*
                }}

                gate_doc!(
                    include => external_doc
                    cfg => doc_cfg
                    masked => doc_masked
                    spotlight => doc_spotlight
                    alias => doc_alias
                    keyword => doc_keyword
                );
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(
                &self,
                non_ascii_idents,
                self.parse_sess.source_map().def_span(sp),
                "non-ascii idents are not fully supported"
            );
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match i.kind {
            ast::ItemKind::ForeignMod(ref foreign_module) => {
                self.check_abi(foreign_module.abi, i.span);
            }

            ast::ItemKind::Fn(..) => {
                if attr::contains_name(&i.attrs[..], sym::plugin_registrar) {
                    gate_feature_post!(&self, plugin_registrar, i.span,
                                       "compiler plugins are experimental and possibly buggy");
                }
                if attr::contains_name(&i.attrs[..], sym::start) {
                    gate_feature_post!(&self, start, i.span,
                                      "a `#[start]` function is an experimental \
                                       feature whose signature may change \
                                       over time");
                }
                if attr::contains_name(&i.attrs[..], sym::main) {
                    gate_feature_post!(&self, main, i.span,
                                       "declaration of a non-standard `#[main]` \
                                        function may change over time, for now \
                                        a top-level `fn main()` is required");
                }
            }

            ast::ItemKind::Struct(..) => {
                for attr in attr::filter_by_name(&i.attrs[..], sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name(sym::simd) {
                            gate_feature_post!(&self, repr_simd, attr.span,
                                               "SIMD types are experimental and possibly buggy");
                        }
                    }
                }
            }

            ast::ItemKind::Enum(ast::EnumDef{ref variants, ..}, ..) => {
                for variant in variants {
                    match (&variant.data, &variant.disr_expr) {
                        (ast::VariantData::Unit(..), _) => {},
                        (_, Some(disr_expr)) =>
                            gate_feature_post!(
                                &self,
                                arbitrary_enum_discriminant,
                                disr_expr.value.span,
                                "discriminants on non-unit variants are experimental"),
                        _ => {},
                    }
                }

                let has_feature = self.features.arbitrary_enum_discriminant;
                if !has_feature && !i.span.allows_unstable(sym::arbitrary_enum_discriminant) {
                    self.maybe_report_invalid_custom_discriminants(&variants);
                }
            }

            ast::ItemKind::Impl(_, polarity, defaultness, ..) => {
                if polarity == ast::ImplPolarity::Negative {
                    gate_feature_post!(&self, optin_builtin_traits,
                                       i.span,
                                       "negative trait bounds are not yet fully implemented; \
                                        use marker types for now");
                }

                if let ast::Defaultness::Default = defaultness {
                    gate_feature_post!(&self, specialization,
                                       i.span,
                                       "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                gate_feature_post!(&self, optin_builtin_traits,
                                   i.span,
                                   "auto traits are experimental and possibly buggy");
            }

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(
                    &self,
                    trait_alias,
                    i.span,
                    "trait aliases are experimental"
                );
            }

            ast::ItemKind::MacroDef(ast::MacroDef { legacy: false, .. }) => {
                let msg = "`macro` is experimental";
                gate_feature_post!(&self, decl_macro, i.span, msg);
            }

            ast::ItemKind::OpaqueTy(..) => {
                gate_feature_post!(
                    &self,
                    type_alias_impl_trait,
                    i.span,
                    "`impl Trait` in type aliases is unstable"
                );
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.kind {
            ast::ForeignItemKind::Fn(..) |
            ast::ForeignItemKind::Static(..) => {
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, sym::link_name);
                let links_to_llvm = match link_name {
                    Some(val) => val.as_str().starts_with("llvm."),
                    _ => false
                };
                if links_to_llvm {
                    gate_feature_post!(&self, link_llvm_intrinsics, i.span,
                                       "linking to LLVM intrinsics is experimental");
                }
            }
            ast::ForeignItemKind::Ty => {
                    gate_feature_post!(&self, extern_types, i.span,
                                       "extern types are experimental");
            }
            ast::ForeignItemKind::Macro(..) => {}
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match ty.kind {
            ast::TyKind::BareFn(ref bare_fn_ty) => {
                self.check_abi(bare_fn_ty.abi, ty.span);
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span,
                                   "The `!` type is experimental");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FunctionRetTy) {
        if let ast::FunctionRetTy::Ty(ref output_ty) = *ret_ty {
            if let ast::TyKind::Never = output_ty.kind {
                // Do nothing.
            } else {
                self.visit_ty(output_ty)
            }
        }
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        match e.kind {
            ast::ExprKind::Box(_) => {
                gate_feature_post!(&self, box_syntax, e.span, EXPLAIN_BOX_SYNTAX);
            }
            ast::ExprKind::Type(..) => {
                // To avoid noise about type ascription in common syntax errors, only emit if it
                // is the *only* error.
                if self.parse_sess.span_diagnostic.err_count() == 0 {
                    gate_feature_post!(&self, type_ascription, e.span,
                                       "type ascription is experimental");
                }
            }
            ast::ExprKind::TryBlock(_) => {
                gate_feature_post!(&self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::Block(_, opt_label) => {
                if let Some(label) = opt_label {
                    gate_feature_post!(&self, label_break_value, label.ident.span,
                                    "labels on blocks are unstable");
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e)
    }

    fn visit_arm(&mut self, arm: &'a ast::Arm) {
        visit::walk_arm(self, arm)
    }

    fn visit_pat(&mut self, pattern: &'a ast::Pat) {
        match &pattern.kind {
            PatKind::Slice(pats) => {
                for pat in &*pats {
                    let span = pat.span;
                    let inner_pat = match &pat.kind {
                        PatKind::Ident(.., Some(pat)) => pat,
                        _ => pat,
                    };
                    if inner_pat.is_rest() {
                        gate_feature_post!(
                            &self,
                            slice_patterns,
                            span,
                            "subslice patterns are unstable"
                        );
                    }
                }
            }
            PatKind::Box(..) => {
                gate_feature_post!(&self, box_patterns,
                                  pattern.span,
                                  "box pattern syntax is experimental");
            }
            PatKind::Range(_, _, Spanned { node: RangeEnd::Excluded, .. }) => {
                gate_feature_post!(&self, exclusive_range_pattern, pattern.span,
                                   "exclusive range pattern syntax is experimental");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(&mut self,
                fn_kind: FnKind<'a>,
                fn_decl: &'a ast::FnDecl,
                span: Span,
                _node_id: NodeId) {
        if let Some(header) = fn_kind.header() {
            // Stability of const fn methods are covered in
            // `visit_trait_item` and `visit_impl_item` below; this is
            // because default methods don't pass through this point.
            self.check_abi(header.abi, span);
        }

        if fn_decl.c_variadic() {
            gate_feature_post!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind, fn_decl, span)
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        match param.kind {
            GenericParamKind::Const { .. } =>
                gate_feature_post!(&self, const_generics, param.ident.span,
                    "const generics are unstable"),
            _ => {}
        }
        visit::walk_generic_param(self, param)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Bound { .. } =>
                gate_feature_post!(&self, associated_type_bounds, constraint.span,
                    "associated type bounds are unstable"),
            _ => {}
        }
        visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.kind {
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_abi(sig.header.abi, ti.span);
                }
                if sig.decl.c_variadic() {
                    gate_feature_post!(&self, c_variadic, ti.span,
                                       "C-variadic functions are unstable");
                }
                if sig.header.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ti.span, "const fn is unstable");
                }
            }
            ast::TraitItemKind::Type(_, ref default) => {
                // We use three if statements instead of something like match guards so that all
                // of these errors can be emitted if all cases apply.
                if default.is_some() {
                    gate_feature_post!(&self, associated_type_defaults, ti.span,
                                       "associated type defaults are unstable");
                }
                if !ti.generics.params.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ti.span,
                                       "generic associated types are unstable");
                }
                if !ti.generics.where_clause.predicates.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ti.span,
                                       "where clauses on associated types are unstable");
                }
            }
            _ => {}
        }
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'a ast::ImplItem) {
        if ii.defaultness == ast::Defaultness::Default {
            gate_feature_post!(&self, specialization,
                              ii.span,
                              "specialization is unstable");
        }

        match ii.kind {
            ast::ImplItemKind::Method(ref sig, _) => {
                if sig.decl.c_variadic() {
                    gate_feature_post!(&self, c_variadic, ii.span,
                                       "C-variadic functions are unstable");
                }
            }
            ast::ImplItemKind::OpaqueTy(..) => {
                gate_feature_post!(
                    &self,
                    type_alias_impl_trait,
                    ii.span,
                    "`impl Trait` in type aliases is unstable"
                );
            }
            ast::ImplItemKind::TyAlias(_) => {
                if !ii.generics.params.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ii.span,
                                       "generic associated types are unstable");
                }
                if !ii.generics.where_clause.predicates.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ii.span,
                                       "where clauses on associated types are unstable");
                }
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii)
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.node {
            gate_feature_post!(&self, crate_visibility_modifier, vis.span,
                               "`crate` visibility modifier is experimental");
        }
        visit::walk_vis(self, vis)
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute],
                    crate_edition: Edition, allow_features: &Option<Vec<String>>) -> Features {
    fn feature_removed(span_handler: &Handler, span: Span, reason: Option<&str>) {
        let mut err = struct_span_err!(span_handler, span, E0557, "feature has been removed");
        if let Some(reason) = reason {
            err.span_note(span, reason);
        } else {
            err.span_label(span, "feature has been removed");
        }
        err.emit();
    }

    let mut features = Features::new();
    let mut edition_enabled_features = FxHashMap::default();

    for &edition in ALL_EDITIONS {
        if edition <= crate_edition {
            // The `crate_edition` implies its respective umbrella feature-gate
            // (i.e., `#![feature(rust_20XX_preview)]` isn't needed on edition 20XX).
            edition_enabled_features.insert(edition.feature_name(), edition);
        }
    }

    for feature in active_features_up_to(crate_edition) {
        feature.set(&mut features, DUMMY_SP);
        edition_enabled_features.insert(feature.name, crate_edition);
    }

    // Process the edition umbrella feature-gates first, to ensure
    // `edition_enabled_features` is completed before it's queried.
    for attr in krate_attrs {
        if !attr.check_name(sym::feature) {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        for mi in list {
            if !mi.is_word() {
                continue;
            }

            let name = mi.name_or_empty();

            let edition = ALL_EDITIONS.iter().find(|e| name == e.feature_name()).copied();
            if let Some(edition) = edition {
                if edition <= crate_edition {
                    continue;
                }

                for feature in active_features_up_to(edition) {
                    // FIXME(Manishearth) there is currently no way to set
                    // lib features by edition
                    feature.set(&mut features, DUMMY_SP);
                    edition_enabled_features.insert(feature.name, edition);
                }
            }
        }
    }

    for attr in krate_attrs {
        if !attr.check_name(sym::feature) {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        let bad_input = |span| {
            struct_span_err!(span_handler, span, E0556, "malformed `feature` attribute input")
        };

        for mi in list {
            let name = match mi.ident() {
                Some(ident) if mi.is_word() => ident.name,
                Some(ident) => {
                    bad_input(mi.span()).span_suggestion(
                        mi.span(),
                        "expected just one word",
                        format!("{}", ident.name),
                        Applicability::MaybeIncorrect,
                    ).emit();
                    continue
                }
                None => {
                    bad_input(mi.span()).span_label(mi.span(), "expected just one word").emit();
                    continue
                }
            };

            if let Some(edition) = edition_enabled_features.get(&name) {
                struct_span_warn!(
                    span_handler,
                    mi.span(),
                    E0705,
                    "the feature `{}` is included in the Rust {} edition",
                    name,
                    edition,
                ).emit();
                continue;
            }

            if ALL_EDITIONS.iter().any(|e| name == e.feature_name()) {
                // Handled in the separate loop above.
                continue;
            }

            let removed = REMOVED_FEATURES.iter().find(|f| name == f.name);
            let stable_removed = STABLE_REMOVED_FEATURES.iter().find(|f| name == f.name);
            if let Some(Feature { state, .. }) = removed.or(stable_removed) {
                if let FeatureState::Removed { reason }
                | FeatureState::Stabilized { reason } = state
                {
                    feature_removed(span_handler, mi.span(), *reason);
                    continue;
                }
            }

            if let Some(Feature { since, .. }) = ACCEPTED_FEATURES.iter().find(|f| name == f.name) {
                let since = Some(Symbol::intern(since));
                features.declared_lang_features.push((name, mi.span(), since));
                continue;
            }

            if let Some(allowed) = allow_features.as_ref() {
                if allowed.iter().find(|&f| f == &name.as_str() as &str).is_none() {
                    span_err!(span_handler, mi.span(), E0725,
                              "the feature `{}` is not in the list of allowed features",
                              name);
                    continue;
                }
            }

            if let Some(f) = ACTIVE_FEATURES.iter().find(|f| name == f.name) {
                f.set(&mut features, mi.span());
                features.declared_lang_features.push((name, mi.span(), None));
                continue;
            }

            features.declared_lib_features.push((name, mi.span()));
        }
    }

    features
}

fn active_features_up_to(edition: Edition) -> impl Iterator<Item=&'static Feature> {
    ACTIVE_FEATURES.iter()
    .filter(move |feature| {
        if let Some(feature_edition) = feature.edition {
            feature_edition <= edition
        } else {
            false
        }
    })
}

pub fn check_crate(krate: &ast::Crate,
                   parse_sess: &ParseSess,
                   features: &Features,
                   unstable: UnstableFeatures) {
    maybe_stage_features(&parse_sess.span_diagnostic, krate, unstable);
    let mut visitor = PostExpansionVisitor { parse_sess, features };

    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => { gate_all!($gate, $gate, $msg); };
        ($spans:ident, $gate:ident, $msg:literal) => {
            for span in &*parse_sess.gated_spans.$spans.borrow() {
                gate_feature!(&visitor, $gate, *span, $msg);
            }
        }
    }

    gate_all!(let_chains, "`let` expressions in this position are experimental");
    gate_all!(async_closure, "async closures are unstable");
    gate_all!(yields, generators, "yield syntax is experimental");
    gate_all!(or_patterns, "or-patterns syntax is experimental");
    gate_all!(const_extern_fn, "`const extern fn` definitions are unstable");

    visit::walk_crate(&mut visitor, krate);
}

#[derive(Clone, Copy, Hash)]
pub enum UnstableFeatures {
    /// Hard errors for unstable features are active, as on beta/stable channels.
    Disallow,
    /// Allow features to be activated, as on nightly.
    Allow,
    /// Errors are bypassed for bootstrapping. This is required any time
    /// during the build that feature-related lints are set to warn or above
    /// because the build turns on warnings-as-errors and uses lots of unstable
    /// features. As a result, this is always required for building Rust itself.
    Cheat
}

impl UnstableFeatures {
    pub fn from_environment() -> UnstableFeatures {
        // `true` if this is a feature-staged build, i.e., on the beta or stable channel.
        let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
        // `true` if we should enable unstable features for bootstrapping.
        let bootstrap = env::var("RUSTC_BOOTSTRAP").is_ok();
        match (disable_unstable_features, bootstrap) {
            (_, true) => UnstableFeatures::Cheat,
            (true, _) => UnstableFeatures::Disallow,
            (false, _) => UnstableFeatures::Allow
        }
    }

    pub fn is_nightly_build(&self) -> bool {
        match *self {
            UnstableFeatures::Allow | UnstableFeatures::Cheat => true,
            UnstableFeatures::Disallow => false,
        }
    }
}

fn maybe_stage_features(span_handler: &Handler, krate: &ast::Crate, unstable: UnstableFeatures) {
    if !unstable.is_nightly_build() {
        for attr in krate.attrs.iter().filter(|attr| attr.check_name(sym::feature)) {
            span_err!(
                span_handler, attr.span, E0554,
                "`#![feature]` may not be used on the {} release channel",
                option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)")
            );
        }
    }
}
