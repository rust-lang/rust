use crate::ast::{self, AssocTyConstraint, AssocTyConstraintKind, NodeId};
use crate::ast::{GenericParam, GenericParamKind, PatKind, RangeEnd, VariantData};
use crate::attr;
use crate::sess::ParseSess;
use crate::visit::{self, FnKind, Visitor};

use rustc_data_structures::fx::FxHashMap;
use rustc_error_codes::*;
use rustc_errors::{error_code, struct_span_err, Applicability, DiagnosticBuilder, Handler};
use rustc_feature::{find_feature_issue, GateIssue};
use rustc_feature::{AttributeGate, BUILTIN_ATTRIBUTE_MAP};
use rustc_feature::{Feature, Features, State as FeatureState, UnstableFeatures};
use rustc_feature::{
    ACCEPTED_FEATURES, ACTIVE_FEATURES, REMOVED_FEATURES, STABLE_REMOVED_FEATURES,
};
use rustc_span::edition::{Edition, ALL_EDITIONS};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{MultiSpan, Span, DUMMY_SP};

use log::debug;

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $level: expr) => {{
        let (cx, has_feature, span, name, explain, level) =
            (&*$cx, $has_feature, $span, $name, $explain, $level);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            leveled_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain, level)
                .emit();
        }
    }};
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!(
            $cx,
            |x: &Features| x.$feature,
            $span,
            sym::$feature,
            $explain,
            GateStrength::Hard
        )
    };
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {
        gate_feature_fn!($cx, |x: &Features| x.$feature, $span, sym::$feature, $explain, $level)
    };
}

pub fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess, features: &Features) {
    PostExpansionVisitor { parse_sess, features }.visit_attribute(attr)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum GateStrength {
    /// A hard error. (Most feature gates should use this.)
    Hard,
    /// Only a warning. (Use this only as backwards-compatibility demands.)
    Soft,
}

pub fn feature_err<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    explain: &str,
) -> DiagnosticBuilder<'a> {
    feature_err_issue(sess, feature, span, GateIssue::Language, explain)
}

pub fn feature_err_issue<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    issue: GateIssue,
    explain: &str,
) -> DiagnosticBuilder<'a> {
    leveled_feature_err(sess, feature, span, issue, explain, GateStrength::Hard)
}

fn leveled_feature_err<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    issue: GateIssue,
    explain: &str,
    level: GateStrength,
) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

    let mut err = match level {
        GateStrength::Hard => diag.struct_span_err_with_code(span, explain, error_code!(E0658)),
        GateStrength::Soft => diag.struct_span_warn(span, explain),
    };

    if let Some(n) = find_feature_issue(feature, issue) {
        err.note(&format!(
            "for more information, see https://github.com/rust-lang/rust/issues/{}",
            n,
        ));
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
    }};
}

impl<'a> PostExpansionVisitor<'a> {
    fn check_abi(&self, abi: ast::StrLit) {
        let ast::StrLit { symbol_unescaped, span, .. } = abi;

        match &*symbol_unescaped.as_str() {
            // Stable
            "Rust" | "C" | "cdecl" | "stdcall" | "fastcall" | "aapcs" | "win64" | "sysv64"
            | "system" => {}
            "rust-intrinsic" => {
                gate_feature_post!(&self, intrinsics, span, "intrinsics are subject to change");
            }
            "platform-intrinsic" => {
                gate_feature_post!(
                    &self,
                    platform_intrinsics,
                    span,
                    "platform intrinsics are experimental and possibly buggy"
                );
            }
            "vectorcall" => {
                gate_feature_post!(
                    &self,
                    abi_vectorcall,
                    span,
                    "vectorcall is experimental and subject to change"
                );
            }
            "thiscall" => {
                gate_feature_post!(
                    &self,
                    abi_thiscall,
                    span,
                    "thiscall is experimental and subject to change"
                );
            }
            "rust-call" => {
                gate_feature_post!(
                    &self,
                    unboxed_closures,
                    span,
                    "rust-call ABI is subject to change"
                );
            }
            "ptx-kernel" => {
                gate_feature_post!(
                    &self,
                    abi_ptx,
                    span,
                    "PTX ABIs are experimental and subject to change"
                );
            }
            "unadjusted" => {
                gate_feature_post!(
                    &self,
                    abi_unadjusted,
                    span,
                    "unadjusted ABI is an implementation detail and perma-unstable"
                );
            }
            "msp430-interrupt" => {
                gate_feature_post!(
                    &self,
                    abi_msp430_interrupt,
                    span,
                    "msp430-interrupt ABI is experimental and subject to change"
                );
            }
            "x86-interrupt" => {
                gate_feature_post!(
                    &self,
                    abi_x86_interrupt,
                    span,
                    "x86-interrupt ABI is experimental and subject to change"
                );
            }
            "amdgpu-kernel" => {
                gate_feature_post!(
                    &self,
                    abi_amdgpu_kernel,
                    span,
                    "amdgpu-kernel ABI is experimental and subject to change"
                );
            }
            "efiapi" => {
                gate_feature_post!(
                    &self,
                    abi_efiapi,
                    span,
                    "efiapi ABI is experimental and subject to change"
                );
            }
            abi => self
                .parse_sess
                .span_diagnostic
                .delay_span_bug(span, &format!("unrecognized ABI not caught in lowering: {}", abi)),
        }
    }

    fn check_extern(&self, ext: ast::Extern) {
        if let ast::Extern::Explicit(abi) = ext {
            self.check_abi(abi);
        }
    }

    fn maybe_report_invalid_custom_discriminants(&self, variants: &[ast::Variant]) {
        let has_fields = variants.iter().any(|variant| match variant.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => true,
            VariantData::Unit(..) => false,
        });

        let discriminant_spans = variants
            .iter()
            .filter(|variant| match variant.data {
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
                "custom discriminant values are not allowed in enums with tuple or struct variants",
            );
            for sp in discriminant_spans {
                err.span_label(sp, "disallowed custom discriminant");
            }
            for variant in variants.iter() {
                match &variant.data {
                    VariantData::Struct(..) => {
                        err.span_label(variant.span, "struct variant defined here");
                    }
                    VariantData::Tuple(..) => {
                        err.span_label(variant.span, "tuple variant defined here");
                    }
                    VariantData::Unit(..) => {}
                }
            }
            err.emit();
        }
    }

    fn check_gat(&self, generics: &ast::Generics, span: Span) {
        if !generics.params.is_empty() {
            gate_feature_post!(
                &self,
                generic_associated_types,
                span,
                "generic associated types are unstable"
            );
        }
        if !generics.where_clause.predicates.is_empty() {
            gate_feature_post!(
                &self,
                generic_associated_types,
                span,
                "where clauses on associated types are unstable"
            );
        }
    }

    /// Feature gate `impl Trait` inside `type Alias = $type_expr;`.
    fn check_impl_trait(&self, ty: &ast::Ty) {
        struct ImplTraitVisitor<'a> {
            vis: &'a PostExpansionVisitor<'a>,
        }
        impl Visitor<'_> for ImplTraitVisitor<'_> {
            fn visit_ty(&mut self, ty: &ast::Ty) {
                if let ast::TyKind::ImplTrait(..) = ty.kind {
                    gate_feature_post!(
                        &self.vis,
                        type_alias_impl_trait,
                        ty.span,
                        "`impl Trait` in type aliases is unstable"
                    );
                }
                visit::walk_ty(self, ty);
            }
        }
        ImplTraitVisitor { vis: self }.visit_ty(ty);
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
                if let Some(abi) = foreign_module.abi {
                    self.check_abi(abi);
                }
            }

            ast::ItemKind::Fn(..) => {
                if attr::contains_name(&i.attrs[..], sym::plugin_registrar) {
                    gate_feature_post!(
                        &self,
                        plugin_registrar,
                        i.span,
                        "compiler plugins are experimental and possibly buggy"
                    );
                }
                if attr::contains_name(&i.attrs[..], sym::start) {
                    gate_feature_post!(
                        &self,
                        start,
                        i.span,
                        "`#[start]` functions are experimental \
                                       and their signature may change \
                                       over time"
                    );
                }
                if attr::contains_name(&i.attrs[..], sym::main) {
                    gate_feature_post!(
                        &self,
                        main,
                        i.span,
                        "declaration of a non-standard `#[main]` \
                                        function may change over time, for now \
                                        a top-level `fn main()` is required"
                    );
                }
            }

            ast::ItemKind::Struct(..) => {
                for attr in attr::filter_by_name(&i.attrs[..], sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name(sym::simd) {
                            gate_feature_post!(
                                &self,
                                repr_simd,
                                attr.span,
                                "SIMD types are experimental and possibly buggy"
                            );
                        }
                    }
                }
            }

            ast::ItemKind::Enum(ast::EnumDef { ref variants, .. }, ..) => {
                for variant in variants {
                    match (&variant.data, &variant.disr_expr) {
                        (ast::VariantData::Unit(..), _) => {}
                        (_, Some(disr_expr)) => gate_feature_post!(
                            &self,
                            arbitrary_enum_discriminant,
                            disr_expr.value.span,
                            "discriminants on non-unit variants are experimental"
                        ),
                        _ => {}
                    }
                }

                let has_feature = self.features.arbitrary_enum_discriminant;
                if !has_feature && !i.span.allows_unstable(sym::arbitrary_enum_discriminant) {
                    self.maybe_report_invalid_custom_discriminants(&variants);
                }
            }

            ast::ItemKind::Impl(_, polarity, defaultness, ..) => {
                if polarity == ast::ImplPolarity::Negative {
                    gate_feature_post!(
                        &self,
                        optin_builtin_traits,
                        i.span,
                        "negative trait bounds are not yet fully implemented; \
                                        use marker types for now"
                    );
                }

                if let ast::Defaultness::Default = defaultness {
                    gate_feature_post!(&self, specialization, i.span, "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                gate_feature_post!(
                    &self,
                    optin_builtin_traits,
                    i.span,
                    "auto traits are experimental and possibly buggy"
                );
            }

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(&self, trait_alias, i.span, "trait aliases are experimental");
            }

            ast::ItemKind::MacroDef(ast::MacroDef { legacy: false, .. }) => {
                let msg = "`macro` is experimental";
                gate_feature_post!(&self, decl_macro, i.span, msg);
            }

            ast::ItemKind::TyAlias(ref ty, ..) => self.check_impl_trait(&ty),

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.kind {
            ast::ForeignItemKind::Fn(..) | ast::ForeignItemKind::Static(..) => {
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, sym::link_name);
                let links_to_llvm = match link_name {
                    Some(val) => val.as_str().starts_with("llvm."),
                    _ => false,
                };
                if links_to_llvm {
                    gate_feature_post!(
                        &self,
                        link_llvm_intrinsics,
                        i.span,
                        "linking to LLVM intrinsics is experimental"
                    );
                }
            }
            ast::ForeignItemKind::Ty => {
                gate_feature_post!(&self, extern_types, i.span, "extern types are experimental");
            }
            ast::ForeignItemKind::Macro(..) => {}
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match ty.kind {
            ast::TyKind::BareFn(ref bare_fn_ty) => {
                self.check_extern(bare_fn_ty.ext);
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span, "The `!` type is experimental");
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
                gate_feature_post!(
                    &self,
                    box_syntax,
                    e.span,
                    "box expression syntax is experimental; you can call `Box::new` instead"
                );
            }
            ast::ExprKind::Type(..) => {
                // To avoid noise about type ascription in common syntax errors, only emit if it
                // is the *only* error.
                if self.parse_sess.span_diagnostic.err_count() == 0 {
                    gate_feature_post!(
                        &self,
                        type_ascription,
                        e.span,
                        "type ascription is experimental"
                    );
                }
            }
            ast::ExprKind::TryBlock(_) => {
                gate_feature_post!(&self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::Block(_, opt_label) => {
                if let Some(label) = opt_label {
                    gate_feature_post!(
                        &self,
                        label_break_value,
                        label.ident.span,
                        "labels on blocks are unstable"
                    );
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
                gate_feature_post!(
                    &self,
                    box_patterns,
                    pattern.span,
                    "box pattern syntax is experimental"
                );
            }
            PatKind::Range(_, _, Spanned { node: RangeEnd::Excluded, .. }) => {
                gate_feature_post!(
                    &self,
                    exclusive_range_pattern,
                    pattern.span,
                    "exclusive range pattern syntax is experimental"
                );
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(
        &mut self,
        fn_kind: FnKind<'a>,
        fn_decl: &'a ast::FnDecl,
        span: Span,
        _node_id: NodeId,
    ) {
        if let Some(header) = fn_kind.header() {
            // Stability of const fn methods are covered in
            // `visit_trait_item` and `visit_impl_item` below; this is
            // because default methods don't pass through this point.
            self.check_extern(header.ext);
        }

        if fn_decl.c_variadic() {
            gate_feature_post!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind, fn_decl, span)
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        match param.kind {
            GenericParamKind::Const { .. } => gate_feature_post!(
                &self,
                const_generics,
                param.ident.span,
                "const generics are unstable"
            ),
            _ => {}
        }
        visit::walk_generic_param(self, param)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Bound { .. } => gate_feature_post!(
                &self,
                associated_type_bounds,
                constraint.span,
                "associated type bounds are unstable"
            ),
            _ => {}
        }
        visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_trait_item(&mut self, ti: &'a ast::AssocItem) {
        match ti.kind {
            ast::AssocItemKind::Fn(ref sig, ref block) => {
                if block.is_none() {
                    self.check_extern(sig.header.ext);
                }
                if sig.header.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ti.span, "const fn is unstable");
                }
            }
            ast::AssocItemKind::TyAlias(_, ref default) => {
                if let Some(_) = default {
                    gate_feature_post!(
                        &self,
                        associated_type_defaults,
                        ti.span,
                        "associated type defaults are unstable"
                    );
                }
            }
            _ => {}
        }
        visit::walk_trait_item(self, ti)
    }

    fn visit_assoc_item(&mut self, ii: &'a ast::AssocItem) {
        if ii.defaultness == ast::Defaultness::Default {
            gate_feature_post!(&self, specialization, ii.span, "specialization is unstable");
        }

        match ii.kind {
            ast::AssocItemKind::Fn(ref sig, _) => {
                if sig.decl.c_variadic() {
                    gate_feature_post!(
                        &self,
                        c_variadic,
                        ii.span,
                        "C-variadic functions are unstable"
                    );
                }
            }
            ast::AssocItemKind::TyAlias(_, ref ty) => {
                if let Some(ty) = ty {
                    self.check_impl_trait(ty);
                }
                self.check_gat(&ii.generics, ii.span);
            }
            _ => {}
        }
        visit::walk_assoc_item(self, ii)
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.node {
            gate_feature_post!(
                &self,
                crate_visibility_modifier,
                vis.span,
                "`crate` visibility modifier is experimental"
            );
        }
        visit::walk_vis(self, vis)
    }
}

pub fn get_features(
    span_handler: &Handler,
    krate_attrs: &[ast::Attribute],
    crate_edition: Edition,
    allow_features: &Option<Vec<String>>,
) -> Features {
    fn feature_removed(span_handler: &Handler, span: Span, reason: Option<&str>) {
        let mut err = struct_span_err!(span_handler, span, E0557, "feature has been removed");
        err.span_label(span, "feature has been removed");
        if let Some(reason) = reason {
            err.note(reason);
        }
        err.emit();
    }

    let mut features = Features::default();
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
            continue;
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
            continue;
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
                    bad_input(mi.span())
                        .span_suggestion(
                            mi.span(),
                            "expected just one word",
                            format!("{}", ident.name),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                    continue;
                }
                None => {
                    bad_input(mi.span()).span_label(mi.span(), "expected just one word").emit();
                    continue;
                }
            };

            if let Some(edition) = edition_enabled_features.get(&name) {
                let msg =
                    &format!("the feature `{}` is included in the Rust {} edition", name, edition);
                span_handler.struct_span_warn_with_code(mi.span(), msg, error_code!(E0705)).emit();
                continue;
            }

            if ALL_EDITIONS.iter().any(|e| name == e.feature_name()) {
                // Handled in the separate loop above.
                continue;
            }

            let removed = REMOVED_FEATURES.iter().find(|f| name == f.name);
            let stable_removed = STABLE_REMOVED_FEATURES.iter().find(|f| name == f.name);
            if let Some(Feature { state, .. }) = removed.or(stable_removed) {
                if let FeatureState::Removed { reason } | FeatureState::Stabilized { reason } =
                    state
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
                if allowed.iter().find(|&f| name.as_str() == *f).is_none() {
                    struct_span_err!(
                        span_handler,
                        mi.span(),
                        E0725,
                        "the feature `{}` is not in the list of allowed features",
                        name
                    )
                    .emit();
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

fn active_features_up_to(edition: Edition) -> impl Iterator<Item = &'static Feature> {
    ACTIVE_FEATURES.iter().filter(move |feature| {
        if let Some(feature_edition) = feature.edition { feature_edition <= edition } else { false }
    })
}

pub fn check_crate(
    krate: &ast::Crate,
    parse_sess: &ParseSess,
    features: &Features,
    unstable: UnstableFeatures,
) {
    maybe_stage_features(&parse_sess.span_diagnostic, krate, unstable);
    let mut visitor = PostExpansionVisitor { parse_sess, features };

    let spans = parse_sess.gated_spans.spans.borrow();
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            for span in spans.get(&sym::$gate).unwrap_or(&vec![]) {
                gate_feature!(&visitor, $gate, *span, $msg);
            }
        };
    }
    gate_all!(let_chains, "`let` expressions in this position are experimental");
    gate_all!(async_closure, "async closures are unstable");
    gate_all!(generators, "yield syntax is experimental");
    gate_all!(or_patterns, "or-patterns syntax is experimental");
    gate_all!(const_extern_fn, "`const extern fn` definitions are unstable");
    gate_all!(raw_ref_op, "raw address of syntax is experimental");
    gate_all!(const_trait_bound_opt_out, "`?const` on trait bounds is experimental");
    gate_all!(const_trait_impl, "const trait impls are experimental");
    gate_all!(half_open_range_patterns, "half-open range patterns are unstable");

    // All uses of `gate_all!` below this point were added in #65742,
    // and subsequently disabled (with the non-early gating readded).
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            // FIXME(eddyb) do something more useful than always
            // disabling these uses of early feature-gatings.
            if false {
                for span in spans.get(&sym::$gate).unwrap_or(&vec![]) {
                    gate_feature!(&visitor, $gate, *span, $msg);
                }
            }
        };
    }

    gate_all!(trait_alias, "trait aliases are experimental");
    gate_all!(associated_type_bounds, "associated type bounds are unstable");
    gate_all!(crate_visibility_modifier, "`crate` visibility modifier is experimental");
    gate_all!(const_generics, "const generics are unstable");
    gate_all!(decl_macro, "`macro` is experimental");
    gate_all!(box_patterns, "box pattern syntax is experimental");
    gate_all!(exclusive_range_pattern, "exclusive range pattern syntax is experimental");
    gate_all!(try_blocks, "`try` blocks are unstable");
    gate_all!(label_break_value, "labels on blocks are unstable");
    gate_all!(box_syntax, "box expression syntax is experimental; you can call `Box::new` instead");
    // To avoid noise about type ascription in common syntax errors,
    // only emit if it is the *only* error. (Also check it last.)
    if parse_sess.span_diagnostic.err_count() == 0 {
        gate_all!(type_ascription, "type ascription is experimental");
    }

    visit::walk_crate(&mut visitor, krate);
}

fn maybe_stage_features(span_handler: &Handler, krate: &ast::Crate, unstable: UnstableFeatures) {
    if !unstable.is_nightly_build() {
        for attr in krate.attrs.iter().filter(|attr| attr.check_name(sym::feature)) {
            struct_span_err!(
                span_handler,
                attr.span,
                E0554,
                "`#![feature]` may not be used on the {} release channel",
                option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)")
            )
            .emit();
        }
    }
}
