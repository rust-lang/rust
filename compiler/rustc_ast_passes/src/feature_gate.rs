use rustc_ast as ast;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::{NodeId, PatKind, attr, token};
use rustc_feature::{AttributeGate, BUILTIN_ATTRIBUTE_MAP, BuiltinAttribute, Features};
use rustc_session::Session;
use rustc_session::parse::{feature_err, feature_warn};
use rustc_span::source_map::Spanned;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::errors;

/// The common case.
macro_rules! gate {
    ($visitor:expr, $feature:ident, $span:expr, $explain:expr) => {{
        if !$visitor.features.$feature() && !$span.allows_unstable(sym::$feature) {
            #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
            feature_err(&$visitor.sess, sym::$feature, $span, $explain).emit();
        }
    }};
    ($visitor:expr, $feature:ident, $span:expr, $explain:expr, $help:expr) => {{
        if !$visitor.features.$feature() && !$span.allows_unstable(sym::$feature) {
            // FIXME: make this translatable
            #[allow(rustc::diagnostic_outside_of_impl)]
            #[allow(rustc::untranslatable_diagnostic)]
            feature_err(&$visitor.sess, sym::$feature, $span, $explain).with_help($help).emit();
        }
    }};
}

/// The unusual case, where the `has_feature` condition is non-standard.
macro_rules! gate_alt {
    ($visitor:expr, $has_feature:expr, $name:expr, $span:expr, $explain:expr) => {{
        if !$has_feature && !$span.allows_unstable($name) {
            #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
            feature_err(&$visitor.sess, $name, $span, $explain).emit();
        }
    }};
    ($visitor:expr, $has_feature:expr, $name:expr, $span:expr, $explain:expr, $notes: expr) => {{
        if !$has_feature && !$span.allows_unstable($name) {
            #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
            let mut diag = feature_err(&$visitor.sess, $name, $span, $explain);
            for note in $notes {
                diag.note(*note);
            }
            diag.emit();
        }
    }};
}

/// The case involving a multispan.
macro_rules! gate_multi {
    ($visitor:expr, $feature:ident, $spans:expr, $explain:expr) => {{
        if !$visitor.features.$feature() {
            let spans: Vec<_> =
                $spans.filter(|span| !span.allows_unstable(sym::$feature)).collect();
            if !spans.is_empty() {
                feature_err(&$visitor.sess, sym::$feature, spans, $explain).emit();
            }
        }
    }};
}

/// The legacy case.
macro_rules! gate_legacy {
    ($visitor:expr, $feature:ident, $span:expr, $explain:expr) => {{
        if !$visitor.features.$feature() && !$span.allows_unstable(sym::$feature) {
            feature_warn(&$visitor.sess, sym::$feature, $span, $explain);
        }
    }};
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
    /// Feature gate `impl Trait` inside `type Alias = $type_expr;`.
    fn check_impl_trait(&self, ty: &ast::Ty, in_associated_ty: bool) {
        struct ImplTraitVisitor<'a> {
            vis: &'a PostExpansionVisitor<'a>,
            in_associated_ty: bool,
        }
        impl Visitor<'_> for ImplTraitVisitor<'_> {
            fn visit_ty(&mut self, ty: &ast::Ty) {
                if let ast::TyKind::ImplTrait(..) = ty.kind {
                    if self.in_associated_ty {
                        gate!(
                            &self.vis,
                            impl_trait_in_assoc_type,
                            ty.span,
                            "`impl Trait` in associated types is unstable"
                        );
                    } else {
                        gate!(
                            &self.vis,
                            type_alias_impl_trait,
                            ty.span,
                            "`impl Trait` in type aliases is unstable"
                        );
                    }
                }
                visit::walk_ty(self, ty);
            }

            fn visit_anon_const(&mut self, _: &ast::AnonConst) -> Self::Result {
                // We don't walk the anon const because it crosses a conceptual boundary: We're no
                // longer "inside" the original type.
                // Brittle: We assume that the callers of `check_impl_trait` will later recurse into
                // the items found in the AnonConst to look for nested TyAliases.
            }
        }
        ImplTraitVisitor { vis: self, in_associated_ty }.visit_ty(ty);
    }

    fn check_late_bound_lifetime_defs(&self, params: &[ast::GenericParam]) {
        // Check only lifetime parameters are present and that the
        // generic parameters that are present have no bounds.
        let non_lt_param_spans = params.iter().filter_map(|param| match param.kind {
            ast::GenericParamKind::Lifetime { .. } => None,
            _ => Some(param.ident.span),
        });
        gate_multi!(
            &self,
            non_lifetime_binders,
            non_lt_param_spans,
            crate::fluent_generated::ast_passes_forbidden_non_lifetime_param
        );

        // FIXME(non_lifetime_binders): Const bound params are pretty broken.
        // Let's keep users from using this feature accidentally.
        if self.features.non_lifetime_binders() {
            let const_param_spans: Vec<_> = params
                .iter()
                .filter_map(|param| match param.kind {
                    ast::GenericParamKind::Const { .. } => Some(param.ident.span),
                    _ => None,
                })
                .collect();

            if !const_param_spans.is_empty() {
                self.sess.dcx().emit_err(errors::ForbiddenConstParam { const_param_spans });
            }
        }

        for param in params {
            if !param.bounds.is_empty() {
                let spans: Vec<_> = param.bounds.iter().map(|b| b.span()).collect();
                self.sess.dcx().emit_err(errors::ForbiddenBound { spans });
            }
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        let attr_info = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));
        // Check feature gates for built-in attributes.
        if let Some(BuiltinAttribute {
            gate: AttributeGate::Gated { feature, message, check, notes, .. },
            ..
        }) = attr_info
        {
            gate_alt!(self, check(self.features), *feature, attr.span, *message, *notes);
        }
        // Check unstable flavors of the `#[doc]` attribute.
        if attr.has_name(sym::doc) {
            for meta_item_inner in attr.meta_item_list().unwrap_or_default() {
                macro_rules! gate_doc { ($($s:literal { $($name:ident => $feature:ident)* })*) => {
                    $($(if meta_item_inner.has_name(sym::$name) {
                        let msg = concat!("`#[doc(", stringify!($name), ")]` is ", $s);
                        gate!(self, $feature, attr.span, msg);
                    })*)*
                }}

                gate_doc!(
                    "experimental" {
                        cfg => doc_cfg
                        cfg_hide => doc_cfg_hide
                        masked => doc_masked
                        notable_trait => doc_notable_trait
                    }
                    "meant for internal use only" {
                        keyword => rustdoc_internals
                        fake_variadic => rustdoc_internals
                        search_unbox => rustdoc_internals
                    }
                );
            }
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match &i.kind {
            ast::ItemKind::ForeignMod(_foreign_module) => {
                // handled during lowering
            }
            ast::ItemKind::Struct(..) | ast::ItemKind::Enum(..) | ast::ItemKind::Union(..) => {
                for attr in attr::filter_by_name(&i.attrs, sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
                        if item.has_name(sym::simd) {
                            gate!(
                                &self,
                                repr_simd,
                                attr.span,
                                "SIMD types are experimental and possibly buggy"
                            );
                        }
                    }
                }
            }

            ast::ItemKind::Impl(box ast::Impl { polarity, defaultness, of_trait, .. }) => {
                if let &ast::ImplPolarity::Negative(span) = polarity {
                    gate!(
                        &self,
                        negative_impls,
                        span.to(of_trait.as_ref().map_or(span, |t| t.path.span)),
                        "negative trait bounds are not fully implemented; \
                         use marker types for now"
                    );
                }

                if let ast::Defaultness::Default(_) = defaultness {
                    gate!(&self, specialization, i.span, "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(box ast::Trait { is_auto: ast::IsAuto::Yes, .. }) => {
                gate!(
                    &self,
                    auto_traits,
                    i.span,
                    "auto traits are experimental and possibly buggy"
                );
            }

            ast::ItemKind::TraitAlias(..) => {
                gate!(&self, trait_alias, i.span, "trait aliases are experimental");
            }

            ast::ItemKind::MacroDef(_, ast::MacroDef { macro_rules: false, .. }) => {
                let msg = "`macro` is experimental";
                gate!(&self, decl_macro, i.span, msg);
            }

            ast::ItemKind::TyAlias(box ast::TyAlias { ty: Some(ty), .. }) => {
                self.check_impl_trait(ty, false)
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.kind {
            ast::ForeignItemKind::Fn(..) | ast::ForeignItemKind::Static(..) => {
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, sym::link_name);
                let links_to_llvm = link_name.is_some_and(|val| val.as_str().starts_with("llvm."));
                if links_to_llvm {
                    gate!(
                        &self,
                        link_llvm_intrinsics,
                        i.span,
                        "linking to LLVM intrinsics is experimental"
                    );
                }
            }
            ast::ForeignItemKind::TyAlias(..) => {
                gate!(&self, extern_types, i.span, "extern types are experimental");
            }
            ast::ForeignItemKind::MacCall(..) => {}
        }

        visit::walk_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match &ty.kind {
            ast::TyKind::BareFn(bare_fn_ty) => {
                // Function pointers cannot be `const`
                self.check_late_bound_lifetime_defs(&bare_fn_ty.generic_params);
            }
            ast::TyKind::Never => {
                gate!(&self, never_type, ty.span, "the `!` type is experimental");
            }
            ast::TyKind::Pat(..) => {
                gate!(&self, pattern_types, ty.span, "pattern types are unstable");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        for predicate in &g.where_clause.predicates {
            match &predicate.kind {
                ast::WherePredicateKind::BoundPredicate(bound_pred) => {
                    // A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
                    self.check_late_bound_lifetime_defs(&bound_pred.bound_generic_params);
                }
                _ => {}
            }
        }
        visit::walk_generics(self, g);
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FnRetTy) {
        if let ast::FnRetTy::Ty(output_ty) = ret_ty {
            if let ast::TyKind::Never = output_ty.kind {
                // Do nothing.
            } else {
                self.visit_ty(output_ty)
            }
        }
    }

    fn visit_generic_args(&mut self, args: &'a ast::GenericArgs) {
        // This check needs to happen here because the never type can be returned from a function,
        // but cannot be used in any other context. If this check was in `visit_fn_ret_ty`, it
        // include both functions and generics like `impl Fn() -> !`.
        if let ast::GenericArgs::Parenthesized(generic_args) = args
            && let ast::FnRetTy::Ty(ref ty) = generic_args.output
            && matches!(ty.kind, ast::TyKind::Never)
        {
            gate!(&self, never_type, ty.span, "the `!` type is experimental");
        }
        visit::walk_generic_args(self, args);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        match e.kind {
            ast::ExprKind::TryBlock(_) => {
                gate!(&self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::Lit(token::Lit {
                kind: token::LitKind::Float | token::LitKind::Integer,
                suffix,
                ..
            }) => match suffix {
                Some(sym::f16) => {
                    gate!(&self, f16, e.span, "the type `f16` is unstable")
                }
                Some(sym::f128) => {
                    gate!(&self, f128, e.span, "the type `f128` is unstable")
                }
                _ => (),
            },
            _ => {}
        }
        visit::walk_expr(self, e)
    }

    fn visit_pat(&mut self, pattern: &'a ast::Pat) {
        match &pattern.kind {
            PatKind::Slice(pats) => {
                for pat in pats {
                    let inner_pat = match &pat.kind {
                        PatKind::Ident(.., Some(pat)) => pat,
                        _ => pat,
                    };
                    if let PatKind::Range(Some(_), None, Spanned { .. }) = inner_pat.kind {
                        gate!(
                            &self,
                            half_open_range_patterns_in_slices,
                            pat.span,
                            "`X..` patterns in slices are experimental"
                        );
                    }
                }
            }
            PatKind::Box(..) => {
                gate!(&self, box_patterns, pattern.span, "box pattern syntax is experimental");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef) {
        self.check_late_bound_lifetime_defs(&t.bound_generic_params);
        visit::walk_poly_trait_ref(self, t);
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, span: Span, _: NodeId) {
        if let Some(_header) = fn_kind.header() {
            // Stability of const fn methods are covered in `visit_assoc_item` below.
        }

        if let FnKind::Closure(ast::ClosureBinder::For { generic_params, .. }, ..) = fn_kind {
            self.check_late_bound_lifetime_defs(generic_params);
        }

        if fn_kind.ctxt() != Some(FnCtxt::Foreign) && fn_kind.decl().c_variadic() {
            gate!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind)
    }

    fn visit_assoc_item(&mut self, i: &'a ast::AssocItem, ctxt: AssocCtxt) {
        let is_fn = match &i.kind {
            ast::AssocItemKind::Fn(_) => true,
            ast::AssocItemKind::Type(box ast::TyAlias { ty, .. }) => {
                if let (Some(_), AssocCtxt::Trait) = (ty, ctxt) {
                    gate!(
                        &self,
                        associated_type_defaults,
                        i.span,
                        "associated type defaults are unstable"
                    );
                }
                if let Some(ty) = ty {
                    self.check_impl_trait(ty, true);
                }
                false
            }
            _ => false,
        };
        if let ast::Defaultness::Default(_) = i.kind.defaultness() {
            // Limit `min_specialization` to only specializing functions.
            gate_alt!(
                &self,
                self.features.specialization() || (is_fn && self.features.min_specialization()),
                sym::specialization,
                i.span,
                "specialization is unstable"
            );
        }
        visit::walk_assoc_item(self, i, ctxt)
    }
}

pub fn check_crate(krate: &ast::Crate, sess: &Session, features: &Features) {
    maybe_stage_features(sess, features, krate);
    check_incompatible_features(sess, features);
    check_new_solver_banned_features(sess, features);

    let mut visitor = PostExpansionVisitor { sess, features };

    let spans = sess.psess.gated_spans.spans.borrow();
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            if let Some(spans) = spans.get(&sym::$gate) {
                for span in spans {
                    gate!(&visitor, $gate, *span, $msg);
                }
            }
        };
        ($gate:ident, $msg:literal, $help:literal) => {
            if let Some(spans) = spans.get(&sym::$gate) {
                for span in spans {
                    gate!(&visitor, $gate, *span, $msg, $help);
                }
            }
        };
    }
    gate_all!(
        if_let_guard,
        "`if let` guards are experimental",
        "you can write `if matches!(<expr>, <pattern>)` instead of `if let <pattern> = <expr>`"
    );
    gate_all!(let_chains, "`let` expressions in this position are unstable");
    gate_all!(
        async_trait_bounds,
        "`async` trait bounds are unstable",
        "use the desugared name of the async trait, such as `AsyncFn`"
    );
    gate_all!(async_for_loop, "`for await` loops are experimental");
    gate_all!(
        closure_lifetime_binder,
        "`for<...>` binders for closures are experimental",
        "consider removing `for<...>`"
    );
    gate_all!(more_qualified_paths, "usage of qualified paths in this context is experimental");
    // yield can be enabled either by `coroutines` or `gen_blocks`
    if let Some(spans) = spans.get(&sym::yield_expr) {
        for span in spans {
            if (!visitor.features.coroutines() && !span.allows_unstable(sym::coroutines))
                && (!visitor.features.gen_blocks() && !span.allows_unstable(sym::gen_blocks))
                && (!visitor.features.yield_expr() && !span.allows_unstable(sym::yield_expr))
            {
                #[allow(rustc::untranslatable_diagnostic)]
                // Emit yield_expr as the error, since that will be sufficient. You can think of it
                // as coroutines and gen_blocks imply yield_expr.
                feature_err(&visitor.sess, sym::yield_expr, *span, "yield syntax is experimental")
                    .emit();
            }
        }
    }
    gate_all!(gen_blocks, "gen blocks are experimental");
    gate_all!(const_trait_impl, "const trait impls are experimental");
    gate_all!(
        half_open_range_patterns_in_slices,
        "half-open range patterns in slices are unstable"
    );
    gate_all!(associated_const_equality, "associated const equality is incomplete");
    gate_all!(yeet_expr, "`do yeet` expression is experimental");
    gate_all!(const_closures, "const closures are experimental");
    gate_all!(builtin_syntax, "`builtin #` syntax is unstable");
    gate_all!(ergonomic_clones, "ergonomic clones are experimental");
    gate_all!(explicit_tail_calls, "`become` expression is experimental");
    gate_all!(generic_const_items, "generic const items are experimental");
    gate_all!(guard_patterns, "guard patterns are experimental", "consider using match arm guards");
    gate_all!(default_field_values, "default values on fields are experimental");
    gate_all!(fn_delegation, "functions delegation is not yet fully implemented");
    gate_all!(postfix_match, "postfix match is experimental");
    gate_all!(mut_ref, "mutable by-reference bindings are experimental");
    gate_all!(global_registration, "global registration is experimental");
    gate_all!(return_type_notation, "return type notation is experimental");
    gate_all!(pin_ergonomics, "pinned reference syntax is experimental");
    gate_all!(unsafe_fields, "`unsafe` fields are experimental");
    gate_all!(unsafe_binders, "unsafe binder types are experimental");
    gate_all!(contracts, "contracts are incomplete");
    gate_all!(contracts_internals, "contract internal machinery is for internal use only");
    gate_all!(where_clause_attrs, "attributes in `where` clause are unstable");
    gate_all!(super_let, "`super let` is experimental");
    gate_all!(frontmatter, "frontmatters are experimental");

    if !visitor.features.never_patterns() {
        if let Some(spans) = spans.get(&sym::never_patterns) {
            for &span in spans {
                if span.allows_unstable(sym::never_patterns) {
                    continue;
                }
                let sm = sess.source_map();
                // We gate two types of spans: the span of a `!` pattern, and the span of a
                // match arm without a body. For the latter we want to give the user a normal
                // error.
                if let Ok(snippet) = sm.span_to_snippet(span)
                    && snippet == "!"
                {
                    #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
                    feature_err(sess, sym::never_patterns, span, "`!` patterns are experimental")
                        .emit();
                } else {
                    let suggestion = span.shrink_to_hi();
                    sess.dcx().emit_err(errors::MatchArmWithNoBody { span, suggestion });
                }
            }
        }
    }

    if !visitor.features.negative_bounds() {
        for &span in spans.get(&sym::negative_bounds).iter().copied().flatten() {
            sess.dcx().emit_err(errors::NegativeBoundUnsupported { span });
        }
    }

    // All uses of `gate_all_legacy_dont_use!` below this point were added in #65742,
    // and subsequently disabled (with the non-early gating readded).
    // We emit an early future-incompatible warning for these.
    // New syntax gates should go above here to get a hard error gate.
    macro_rules! gate_all_legacy_dont_use {
        ($gate:ident, $msg:literal) => {
            for span in spans.get(&sym::$gate).unwrap_or(&vec![]) {
                gate_legacy!(&visitor, $gate, *span, $msg);
            }
        };
    }

    gate_all_legacy_dont_use!(box_patterns, "box pattern syntax is experimental");
    gate_all_legacy_dont_use!(trait_alias, "trait aliases are experimental");
    gate_all_legacy_dont_use!(decl_macro, "`macro` is experimental");
    gate_all_legacy_dont_use!(try_blocks, "`try` blocks are unstable");
    gate_all_legacy_dont_use!(auto_traits, "`auto` traits are unstable");

    visit::walk_crate(&mut visitor, krate);
}

fn maybe_stage_features(sess: &Session, features: &Features, krate: &ast::Crate) {
    // checks if `#![feature]` has been used to enable any feature.
    if sess.opts.unstable_features.is_nightly_build() {
        return;
    }
    if features.enabled_features().is_empty() {
        return;
    }
    let mut errored = false;
    for attr in krate.attrs.iter().filter(|attr| attr.has_name(sym::feature)) {
        // `feature(...)` used on non-nightly. This is definitely an error.
        let mut err = errors::FeatureOnNonNightly {
            span: attr.span,
            channel: option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)"),
            stable_features: vec![],
            sugg: None,
        };

        let mut all_stable = true;
        for ident in attr.meta_item_list().into_iter().flatten().flat_map(|nested| nested.ident()) {
            let name = ident.name;
            let stable_since = features
                .enabled_lang_features()
                .iter()
                .find(|feat| feat.gate_name == name)
                .map(|feat| feat.stable_since)
                .flatten();
            if let Some(since) = stable_since {
                err.stable_features.push(errors::StableFeature { name, since });
            } else {
                all_stable = false;
            }
        }
        if all_stable {
            err.sugg = Some(attr.span);
        }
        sess.dcx().emit_err(err);
        errored = true;
    }
    // Just make sure we actually error if anything is listed in `enabled_features`.
    assert!(errored);
}

fn check_incompatible_features(sess: &Session, features: &Features) {
    let enabled_lang_features =
        features.enabled_lang_features().iter().map(|feat| (feat.gate_name, feat.attr_sp));
    let enabled_lib_features =
        features.enabled_lib_features().iter().map(|feat| (feat.gate_name, feat.attr_sp));
    let enabled_features = enabled_lang_features.chain(enabled_lib_features);

    for (f1, f2) in rustc_feature::INCOMPATIBLE_FEATURES
        .iter()
        .filter(|(f1, f2)| features.enabled(*f1) && features.enabled(*f2))
    {
        if let Some((f1_name, f1_span)) = enabled_features.clone().find(|(name, _)| name == f1) {
            if let Some((f2_name, f2_span)) = enabled_features.clone().find(|(name, _)| name == f2)
            {
                let spans = vec![f1_span, f2_span];
                sess.dcx().emit_err(errors::IncompatibleFeatures {
                    spans,
                    f1: f1_name,
                    f2: f2_name,
                });
            }
        }
    }
}

fn check_new_solver_banned_features(sess: &Session, features: &Features) {
    if !sess.opts.unstable_opts.next_solver.globally {
        return;
    }

    // Ban GCE with the new solver, because it does not implement GCE correctly.
    if let Some(gce_span) = features
        .enabled_lang_features()
        .iter()
        .find(|feat| feat.gate_name == sym::generic_const_exprs)
        .map(|feat| feat.attr_sp)
    {
        #[allow(rustc::symbol_intern_string_literal)]
        sess.dcx().emit_err(errors::IncompatibleFeatures {
            spans: vec![gce_span],
            f1: Symbol::intern("-Znext-solver=globally"),
            f2: sym::generic_const_exprs,
        });
    }
}
