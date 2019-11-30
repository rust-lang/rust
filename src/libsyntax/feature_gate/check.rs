use rustc_feature::{ACCEPTED_FEATURES, ACTIVE_FEATURES, REMOVED_FEATURES, STABLE_REMOVED_FEATURES};
use rustc_feature::{AttributeGate, BUILTIN_ATTRIBUTE_MAP};
use rustc_feature::{Features, Feature, State as FeatureState, UnstableFeatures};
use rustc_feature::{find_feature_issue, GateIssue};

use crate::ast::{self, AssocTyConstraint, AssocTyConstraintKind, NodeId};
use crate::ast::{GenericParam, GenericParamKind, PatKind, RangeEnd, VariantData};
use crate::attr;
use crate::visit::{self, FnKind, Visitor};
use crate::sess::ParseSess;

use errors::{Applicability, DiagnosticBuilder, Handler};
use rustc_data_structures::fx::FxHashMap;
use syntax_pos::{Span, DUMMY_SP, MultiSpan};
use syntax_pos::edition::{ALL_EDITIONS, Edition};
use syntax_pos::source_map::Spanned;
use syntax_pos::symbol::{Symbol, sym};
use log::debug;

use rustc_error_codes::*;

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
        GateStrength::Hard => {
            diag.struct_span_err_with_code(span, explain, stringify_error_code!(E0658))
        }
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

pub fn gate_feature(
    parse_sess: &ParseSess,
    features: &Features,
    span: Span,
    feature: Symbol,
    explain: &str,
) {
    PostExpansionVisitor { parse_sess, features }.gate(span, feature, explain)
}

struct PostExpansionVisitor<'a> {
    parse_sess: &'a ParseSess,
    features: &'a Features,
}

impl<'a> PostExpansionVisitor<'a> {
    fn gate(&self, span: Span, feature: Symbol, explain: &str) {
        if !span.allows_unstable(feature) {
            let has: bool = self.features.on(feature);
            debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", feature, span, has);
            if !has && !span.allows_unstable(feature) {
               feature_err(self.parse_sess, feature, span, explain).emit();
            }
        }
    }

    fn check_abi(&self, abi: ast::StrLit) {
        let ast::StrLit { symbol_unescaped, span, .. } = abi;

        let (feature, explain) = match &*symbol_unescaped.as_str() {
            // Stable
            "Rust" |
            "C" |
            "cdecl" |
            "stdcall" |
            "fastcall" |
            "aapcs" |
            "win64" |
            "sysv64" |
            "system" => return,
            "rust-intrinsic" => (
                sym::intrinsics,
                "intrinsics are subject to change",
            ),
            "platform-intrinsic" => (
                sym::platform_intrinsics,
                "platform intrinsics are experimental and possibly buggy",
            ),
            "vectorcall" => (
                sym::abi_vectorcall,
                "vectorcall is experimental and subject to change",
            ),
            "thiscall" => (
                sym::abi_thiscall,
                "thiscall is experimental and subject to change",
            ),
            "rust-call" => (
                sym::unboxed_closures,
                "rust-call ABI is subject to change",
            ),
            "ptx-kernel" => (
                sym::abi_ptx,
                "PTX ABIs are experimental and subject to change",
            ),
            "unadjusted" => (
                sym::abi_unadjusted,
                "unadjusted ABI is an implementation detail and perma-unstable",
            ),
            "msp430-interrupt" => (
                sym::abi_msp430_interrupt,
                "msp430-interrupt ABI is experimental and subject to change",
            ),
            "x86-interrupt" => (
                sym::abi_x86_interrupt,
                "x86-interrupt ABI is experimental and subject to change",
            ),
            "amdgpu-kernel" => (
                sym::abi_amdgpu_kernel,
                "amdgpu-kernel ABI is experimental and subject to change",
            ),
            "efiapi" => (
                sym::abi_efiapi,
                "efiapi ABI is experimental and subject to change",
            ),
            abi => {
                self.parse_sess.span_diagnostic.delay_span_bug(
                    span,
                    &format!("unrecognized ABI not caught in lowering: {}", abi),
                );
                return;
            }
        };
        self.gate(span, feature, explain);
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

    fn check_gat(&self, generics: &ast::Generics, span: Span) {
        if !generics.params.is_empty() {
            self.gate(span, sym::generic_associated_types, "generic associated types are unstable");
        }
        if !generics.where_clause.predicates.is_empty() {
            self.gate(
                span, sym::generic_associated_types,
                "where clauses on associated types are unstable",
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
                    self.vis.gate(
                        ty.span, sym::type_alias_impl_trait,
                        "`impl Trait` in type aliases is unstable",
                    );
                }
                visit::walk_ty(self, ty);
            }
        }
        ImplTraitVisitor { vis: self }.visit_ty(ty);
    }

    fn check_c_variadic(&self, span: Span, decl: &ast::FnDecl) {
        if decl.c_variadic() {
            self.gate(span, sym::c_variadic, "C-variadic functions are unstable");
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        let attr_info =
            attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name)).map(|a| **a);
        // Check feature gates for built-in attributes.
        if let Some((.., AttributeGate::Gated(_, feature, descr))) = attr_info {
            self.gate(attr.span, feature, descr);
        }
        // Check unstable flavors of the `#[doc]` attribute.
        if attr.check_name(sym::doc) {
            for nested_meta in attr.meta_item_list().unwrap_or_default() {
                const GATED_DOC_FEATURES: &[(Symbol, Symbol, &str)] = &[
                    (sym::include, sym::external_doc, "`#[doc(include)]` is experimental"),
                    (sym::cfg, sym::doc_cfg, "`#[doc(cfg)]` is experimental"),
                    (sym::masked, sym::doc_masked, "`#[doc(masked)]` is experimental"),
                    (sym::spotlight, sym::doc_spotlight, "`#[doc(spotlight)]` is experimental"),
                    (sym::alias, sym::doc_alias, "`#[doc(alias)]` is experimental"),
                    (sym::keyword, sym::doc_keyword, "`#[doc(keyword)]` is experimental"),
                ];
                for (name, feature, explain) in GATED_DOC_FEATURES {
                    if nested_meta.check_name(*name) {
                        self.gate(attr.span, *feature, explain);
                    }
                }
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            self.gate(
                self.parse_sess.source_map().def_span(sp), sym::non_ascii_idents,
                "non-ascii idents are not fully supported",
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
                    self.gate(
                        i.span, sym::plugin_registrar,
                        "compiler plugins are experimental and possibly buggy",
                    );
                }
                if attr::contains_name(&i.attrs[..], sym::start) {
                    self.gate(
                        i.span, sym::start,
                        "a `#[start]` function is an experimental \
                        feature whose signature may change over time",
                    );
                }
                if attr::contains_name(&i.attrs[..], sym::main) {
                    self.gate(
                        i.span, sym::main,
                        "declaration of a non-standard `#[main]` function may change over time, \
                        for now a top-level `fn main()` is required",
                    );
                }
            }

            ast::ItemKind::Struct(..) => {
                for attr in attr::filter_by_name(&i.attrs[..], sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name(sym::simd) {
                            self.gate(
                                attr.span, sym::repr_simd,
                                "SIMD types are experimental and possibly buggy",
                            );
                        }
                    }
                }
            }

            ast::ItemKind::Enum(ast::EnumDef{ref variants, ..}, ..) => {
                for variant in variants {
                    match (&variant.data, &variant.disr_expr) {
                        (ast::VariantData::Unit(..), _) => {},
                        (_, Some(disr_expr)) => self.gate(
                            disr_expr.value.span, sym::arbitrary_enum_discriminant,
                            "discriminants on non-unit variants are experimental",
                        ),
                        _ => {},
                    }
                }

                let has_feature = self.features.on(sym::arbitrary_enum_discriminant);
                if !has_feature && !i.span.allows_unstable(sym::arbitrary_enum_discriminant) {
                    self.maybe_report_invalid_custom_discriminants(&variants);
                }
            }

            ast::ItemKind::Impl(_, polarity, defaultness, ..) => {
                if polarity == ast::ImplPolarity::Negative {
                    self.gate(
                        i.span, sym::optin_builtin_traits,
                        "negative trait bounds are not yet fully implemented; \
                        use marker types for now",
                    );
                }

                if let ast::Defaultness::Default = defaultness {
                    self.gate(i.span, sym::specialization, "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                self.gate(
                    i.span, sym::optin_builtin_traits,
                    "auto traits are experimental and possibly buggy",
                );
            }

            ast::ItemKind::TraitAlias(..) => {
                self.gate(i.span, sym::trait_alias, "trait aliases are experimental");
            }

            ast::ItemKind::MacroDef(ast::MacroDef { legacy: false, .. }) => {
                self.gate(i.span, sym::decl_macro, "`macro` is experimental");
            }

            ast::ItemKind::TyAlias(ref ty, ..) => self.check_impl_trait(&ty),

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
                    self.gate(
                        i.span, sym::link_llvm_intrinsics,
                        "linking to LLVM intrinsics is experimental",
                    );
                }
            }
            ast::ForeignItemKind::Ty => {
                self.gate(i.span, sym::extern_types, "extern types are experimental");
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
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        match e.kind {
            ast::ExprKind::Box(_) => {
                self.gate(
                    e.span, sym::box_syntax,
                    "box expression syntax is experimental; you can call `Box::new` instead",
                );
            }
            ast::ExprKind::Type(..) => {
                // To avoid noise about type ascription in common syntax errors, only emit if it
                // is the *only* error.
                if self.parse_sess.span_diagnostic.err_count() == 0 {
                    self.gate(e.span, sym::type_ascription, "type ascription is experimental");
                }
            }
            ast::ExprKind::TryBlock(_) => {
                self.gate(e.span, sym::try_blocks, "`try` expression is experimental");
            }
            ast::ExprKind::Block(_, opt_label) => {
                if let Some(label) = opt_label {
                    self.gate(
                        label.ident.span, sym::label_break_value,
                        "labels on blocks are unstable",
                    );
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e)
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
                        self.gate(span, sym::slice_patterns, "subslice patterns are unstable");
                    }
                }
            }
            PatKind::Box(..) => {
                self.gate(pattern.span, sym::box_patterns, "box pattern syntax is experimental");
            }
            PatKind::Range(_, _, Spanned { node: RangeEnd::Excluded, .. }) => {
                self.gate(
                    pattern.span, sym::exclusive_range_pattern,
                    "exclusive range pattern syntax is experimental",
                );
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
            self.check_extern(header.ext);
        }

        self.check_c_variadic(span, fn_decl);

        visit::walk_fn(self, fn_kind, fn_decl, span)
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        match param.kind {
            GenericParamKind::Const { .. } => {
                self.gate(param.ident.span, sym::const_generics, "const generics are unstable");
            }
            _ => {}
        }
        visit::walk_generic_param(self, param)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Bound { .. } => {
                self.gate(
                    constraint.span, sym::associated_type_bounds,
                    "associated type bounds are unstable",
                );
            }
            _ => {}
        }
        visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.kind {
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_extern(sig.header.ext);
                }
                self.check_c_variadic(ti.span, &sig.decl);
                if sig.header.constness.node == ast::Constness::Const {
                    self.gate(ti.span, sym::const_fn, "const fn is unstable");
                }
            }
            ast::TraitItemKind::Type(_, ref default) => {
                if let Some(ty) = default {
                    self.check_impl_trait(ty);
                    self.gate(
                        ti.span, sym::associated_type_defaults,
                        "associated type defaults are unstable",
                    );
                }
                self.check_gat(&ti.generics, ti.span);
            }
            _ => {}
        }
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'a ast::ImplItem) {
        if ii.defaultness == ast::Defaultness::Default {
            self.gate(ii.span, sym::specialization, "specialization is unstable");
        }

        match ii.kind {
            ast::ImplItemKind::Method(ref sig, _) => {
                self.check_c_variadic(ii.span, &sig.decl);
            },
            ast::ImplItemKind::TyAlias(ref ty) => {
                self.check_impl_trait(ty);
                self.check_gat(&ii.generics, ii.span);
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii)
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.node {
            self.gate(
                vis.span, sym::crate_visibility_modifier,
                "`crate` visibility modifier is experimental",
            );
        }
        visit::walk_vis(self, vis)
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute],
                    crate_edition: Edition, allow_features: &Option<Vec<String>>) -> Features {
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
        features.enable(feature, DUMMY_SP);
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
                    features.enable(feature, DUMMY_SP);
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
                if allowed.iter().find(|&f| name.as_str() == *f).is_none() {
                    span_err!(span_handler, mi.span(), E0725,
                              "the feature `{}` is not in the list of allowed features",
                              name);
                    continue;
                }
            }

            if let Some(f) = ACTIVE_FEATURES.iter().find(|f| name == f.name) {
                features.enable(f, mi.span());
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

    let spans = parse_sess.gated_spans.spans.borrow();
    let gate_all = |gate, msg| {
        for span in spans.get(&gate).unwrap_or(&vec![]) {
            visitor.gate(*span, gate, msg);
        }
    };

    gate_all(sym::let_chains, "`let` expressions in this position are experimental");
    gate_all(sym::async_closure, "async closures are unstable");
    gate_all(sym::generators, "yield syntax is experimental");
    gate_all(sym::or_patterns, "or-patterns syntax is experimental");
    gate_all(sym::const_extern_fn, "`const extern fn` definitions are unstable");
    gate_all(sym::raw_ref_op, "raw address of syntax is experimental");

    // All uses of `gate_all!` below this point were added in #65742,
    // and subsequently disabled (with the non-early gating readded).
    let gate_all = |gate, msg| {
        // FIXME(eddyb) do something more useful than always
        // disabling these uses of early feature-gatings.
        if false { gate_all(gate, msg); }
    };

    gate_all(sym::trait_alias, "trait aliases are experimental");
    gate_all(sym::associated_type_bounds, "associated type bounds are unstable");
    gate_all(sym::crate_visibility_modifier, "`crate` visibility modifier is experimental");
    gate_all(sym::const_generics, "const generics are unstable");
    gate_all(sym::decl_macro, "`macro` is experimental");
    gate_all(sym::box_patterns, "box pattern syntax is experimental");
    gate_all(sym::exclusive_range_pattern, "exclusive range pattern syntax is experimental");
    gate_all(sym::try_blocks, "`try` blocks are unstable");
    gate_all(sym::label_break_value, "labels on blocks are unstable");
    gate_all(
        sym::box_syntax,
        "box expression syntax is experimental; you can call `Box::new` instead",
    );
    // To avoid noise about type ascription in common syntax errors,
    // only emit if it is the *only* error. (Also check it last.)
    if parse_sess.span_diagnostic.err_count() == 0 {
        gate_all(sym::type_ascription, "type ascription is experimental");
    }

    visit::walk_crate(&mut visitor, krate);
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
