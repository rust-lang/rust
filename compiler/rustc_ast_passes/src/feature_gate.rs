use rustc_ast as ast;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::{AssocTyConstraint, AssocTyConstraintKind, NodeId};
use rustc_ast::{PatKind, RangeEnd, VariantData};
use rustc_errors::struct_span_err;
use rustc_feature::{AttributeGate, BUILTIN_ATTRIBUTE_MAP};
use rustc_feature::{Features, GateIssue};
use rustc_session::parse::{feature_err, feature_err_issue};
use rustc_session::Session;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use tracing::debug;

macro_rules! gate_feature_fn {
    ($visitor: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr) => {{
        let (visitor, has_feature, span, name, explain) =
            (&*$visitor, $has_feature, $span, $name, $explain);
        let has_feature: bool = has_feature(visitor.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            feature_err_issue(&visitor.sess.parse_sess, name, span, GateIssue::Language, explain)
                .emit();
        }
    }};
}

macro_rules! gate_feature_post {
    ($visitor: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($visitor, |x: &Features| x.$feature, $span, sym::$feature, $explain)
    };
}

pub fn check_attribute(attr: &ast::Attribute, sess: &Session, features: &Features) {
    PostExpansionVisitor { sess, features }.visit_attribute(attr)
}

struct PostExpansionVisitor<'a> {
    sess: &'a Session,

    // `sess` contains a `Features`, but this might not be that one.
    features: &'a Features,
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
            "avr-interrupt" | "avr-non-blocking-interrupt" => {
                gate_feature_post!(
                    &self,
                    abi_avr_interrupt,
                    span,
                    "avr-interrupt and avr-non-blocking-interrupt ABIs are experimental and subject to change"
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
                .sess
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
                &self.sess.parse_sess,
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
            gate_feature_fn!(self, has_feature, attr.span, name, descr);
        }
        // Check unstable flavors of the `#[doc]` attribute.
        if self.sess.check_name(attr, sym::doc) {
            for nested_meta in attr.meta_item_list().unwrap_or_default() {
                macro_rules! gate_doc { ($($name:ident => $feature:ident)*) => {
                    $(if nested_meta.has_name(sym::$name) {
                        let msg = concat!("`#[doc(", stringify!($name), ")]` is experimental");
                        gate_feature_post!(self, $feature, attr.span, msg);
                    })*
                }}

                gate_doc!(
                    include => external_doc
                    cfg => doc_cfg
                    masked => doc_masked
                    spotlight => doc_spotlight
                    keyword => doc_keyword
                );
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: Symbol) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(
                &self,
                non_ascii_idents,
                self.sess.parse_sess.source_map().guess_head_span(sp),
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
                if self.sess.contains_name(&i.attrs[..], sym::plugin_registrar) {
                    gate_feature_post!(
                        &self,
                        plugin_registrar,
                        i.span,
                        "compiler plugins are experimental and possibly buggy"
                    );
                }
                if self.sess.contains_name(&i.attrs[..], sym::start) {
                    gate_feature_post!(
                        &self,
                        start,
                        i.span,
                        "`#[start]` functions are experimental \
                         and their signature may change \
                         over time"
                    );
                }
                if self.sess.contains_name(&i.attrs[..], sym::main) {
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
                for attr in self.sess.filter_by_name(&i.attrs[..], sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.has_name(sym::simd) {
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

            ast::ItemKind::Impl { polarity, defaultness, ref of_trait, .. } => {
                if let ast::ImplPolarity::Negative(span) = polarity {
                    gate_feature_post!(
                        &self,
                        negative_impls,
                        span.to(of_trait.as_ref().map(|t| t.path.span).unwrap_or(span)),
                        "negative trait bounds are not yet fully implemented; \
                         use marker types for now"
                    );
                }

                if let ast::Defaultness::Default(_) = defaultness {
                    gate_feature_post!(&self, specialization, i.span, "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                gate_feature_post!(
                    &self,
                    auto_traits,
                    i.span,
                    "auto traits are experimental and possibly buggy"
                );
            }

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(&self, trait_alias, i.span, "trait aliases are experimental");
            }

            ast::ItemKind::MacroDef(ast::MacroDef { macro_rules: false, .. }) => {
                let msg = "`macro` is experimental";
                gate_feature_post!(&self, decl_macro, i.span, msg);
            }

            ast::ItemKind::TyAlias(_, _, _, Some(ref ty)) => self.check_impl_trait(&ty),

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.kind {
            ast::ForeignItemKind::Fn(..) | ast::ForeignItemKind::Static(..) => {
                let link_name = self.sess.first_attr_value_str_by_name(&i.attrs, sym::link_name);
                let links_to_llvm =
                    link_name.map_or(false, |val| val.as_str().starts_with("llvm."));
                if links_to_llvm {
                    gate_feature_post!(
                        &self,
                        link_llvm_intrinsics,
                        i.span,
                        "linking to LLVM intrinsics is experimental"
                    );
                }
            }
            ast::ForeignItemKind::TyAlias(..) => {
                gate_feature_post!(&self, extern_types, i.span, "extern types are experimental");
            }
            ast::ForeignItemKind::MacCall(..) => {}
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match ty.kind {
            ast::TyKind::BareFn(ref bare_fn_ty) => {
                self.check_extern(bare_fn_ty.ext);
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span, "the `!` type is experimental");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FnRetTy) {
        if let ast::FnRetTy::Ty(ref output_ty) = *ret_ty {
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
                if self.sess.parse_sess.span_diagnostic.err_count() == 0 {
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

    fn visit_pat(&mut self, pattern: &'a ast::Pat) {
        match &pattern.kind {
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

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, span: Span, _: NodeId) {
        if let Some(header) = fn_kind.header() {
            // Stability of const fn methods are covered in `visit_assoc_item` below.
            self.check_extern(header.ext);

            if let (ast::Const::Yes(_), ast::Extern::Implicit)
            | (ast::Const::Yes(_), ast::Extern::Explicit(_)) = (header.constness, header.ext)
            {
                gate_feature_post!(
                    &self,
                    const_extern_fn,
                    span,
                    "`const extern fn` definitions are unstable"
                );
            }
        }

        if fn_kind.ctxt() != Some(FnCtxt::Foreign) && fn_kind.decl().c_variadic() {
            gate_feature_post!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind, span)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'a AssocTyConstraint) {
        if let AssocTyConstraintKind::Bound { .. } = constraint.kind {
            gate_feature_post!(
                &self,
                associated_type_bounds,
                constraint.span,
                "associated type bounds are unstable"
            )
        }
        visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_assoc_item(&mut self, i: &'a ast::AssocItem, ctxt: AssocCtxt) {
        let is_fn = match i.kind {
            ast::AssocItemKind::Fn(_, ref sig, _, _) => {
                if let (ast::Const::Yes(_), AssocCtxt::Trait) = (sig.header.constness, ctxt) {
                    gate_feature_post!(&self, const_fn, i.span, "const fn is unstable");
                }
                true
            }
            ast::AssocItemKind::TyAlias(_, ref generics, _, ref ty) => {
                if let (Some(_), AssocCtxt::Trait) = (ty, ctxt) {
                    gate_feature_post!(
                        &self,
                        associated_type_defaults,
                        i.span,
                        "associated type defaults are unstable"
                    );
                }
                if let Some(ty) = ty {
                    self.check_impl_trait(ty);
                }
                self.check_gat(generics, i.span);
                false
            }
            _ => false,
        };
        if let ast::Defaultness::Default(_) = i.kind.defaultness() {
            // Limit `min_specialization` to only specializing functions.
            gate_feature_fn!(
                &self,
                |x: &Features| x.specialization || (is_fn && x.min_specialization),
                i.span,
                sym::specialization,
                "specialization is unstable"
            );
        }
        visit::walk_assoc_item(self, i, ctxt)
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.kind {
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

pub fn check_crate(krate: &ast::Crate, sess: &Session) {
    maybe_stage_features(sess, krate);
    check_incompatible_features(sess);
    let mut visitor = PostExpansionVisitor { sess, features: &sess.features_untracked() };

    let spans = sess.parse_sess.gated_spans.spans.borrow();
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            if let Some(spans) = spans.get(&sym::$gate) {
                for span in spans {
                    gate_feature_post!(&visitor, $gate, *span, $msg);
                }
            }
        };
    }
    gate_all!(if_let_guard, "`if let` guards are experimental");
    gate_all!(let_chains, "`let` expressions in this position are experimental");
    gate_all!(async_closure, "async closures are unstable");
    gate_all!(generators, "yield syntax is experimental");
    gate_all!(or_patterns, "or-patterns syntax is experimental");
    gate_all!(raw_ref_op, "raw address of syntax is experimental");
    gate_all!(const_trait_bound_opt_out, "`?const` on trait bounds is experimental");
    gate_all!(const_trait_impl, "const trait impls are experimental");
    gate_all!(half_open_range_patterns, "half-open range patterns are unstable");
    gate_all!(inline_const, "inline-const is experimental");
    gate_all!(
        extended_key_value_attributes,
        "arbitrary expressions in key-value attributes are unstable"
    );
    if sess.parse_sess.span_diagnostic.err_count() == 0 {
        // Errors for `destructuring_assignment` can get quite noisy, especially where `_` is
        // involved, so we only emit errors where there are no other parsing errors.
        gate_all!(destructuring_assignment, "destructuring assignments are unstable");
    }

    // All uses of `gate_all!` below this point were added in #65742,
    // and subsequently disabled (with the non-early gating readded).
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            // FIXME(eddyb) do something more useful than always
            // disabling these uses of early feature-gatings.
            if false {
                for span in spans.get(&sym::$gate).unwrap_or(&vec![]) {
                    gate_feature_post!(&visitor, $gate, *span, $msg);
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
    if sess.parse_sess.span_diagnostic.err_count() == 0 {
        gate_all!(type_ascription, "type ascription is experimental");
    }

    visit::walk_crate(&mut visitor, krate);
}

fn maybe_stage_features(sess: &Session, krate: &ast::Crate) {
    if !sess.opts.unstable_features.is_nightly_build() {
        for attr in krate.attrs.iter().filter(|attr| sess.check_name(attr, sym::feature)) {
            struct_span_err!(
                sess.parse_sess.span_diagnostic,
                attr.span,
                E0554,
                "`#![feature]` may not be used on the {} release channel",
                option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)")
            )
            .emit();
        }
    }
}

fn check_incompatible_features(sess: &Session) {
    let features = sess.features_untracked();

    let declared_features = features
        .declared_lang_features
        .iter()
        .copied()
        .map(|(name, span, _)| (name, span))
        .chain(features.declared_lib_features.iter().copied());

    for (f1, f2) in rustc_feature::INCOMPATIBLE_FEATURES
        .iter()
        .filter(|&&(f1, f2)| features.enabled(f1) && features.enabled(f2))
    {
        if let Some((f1_name, f1_span)) = declared_features.clone().find(|(name, _)| name == f1) {
            if let Some((f2_name, f2_span)) = declared_features.clone().find(|(name, _)| name == f2)
            {
                let spans = vec![f1_span, f2_span];
                sess.struct_span_err(
                    spans.clone(),
                    &format!(
                        "features `{}` and `{}` are incompatible, using them at the same time \
                        is not allowed",
                        f1_name, f2_name
                    ),
                )
                .help("remove one of these features")
                .emit();
            }
        }
    }
}
