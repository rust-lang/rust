use rustc_ast::visit::{self, AssocCtxt, FnKind, Visitor};
use rustc_ast::{
    self as ast, AttrVec, BindingMode, ByRef, GenericBound, GenericParamKind, NodeId, PatKind,
    attr, token,
};
use rustc_ast_pretty::pprust;
use rustc_attr_ir::{Attribute, AttributeKind};
use rustc_attr_parsing::AttributeParser;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::msg;
use rustc_feature::Features;
use rustc_session::Session;
use rustc_session::diagnostics::{feature_err, feature_warn};
use rustc_span::{Ident, Span, Spanned, Symbol, sym};

use crate::diagnostics;

/// The common case.
macro_rules! gate {
    ($visitor:expr, $feature:ident, $span:expr, $explain:expr $(, $help:expr)?) => {{
        if !$visitor.features.$feature() && !$span.allows_unstable(sym::$feature) {
            feature_err($visitor.sess, sym::$feature, $span, $explain)
                $(.with_help($help))?
                .emit();
        }
    }};
}

/// The unusual case, where the `has_feature` condition is non-standard.
macro_rules! gate_alt {
    ($visitor:expr, $has_feature:expr, $name:expr, $span:expr, $explain:expr $(, $notes:expr)?) => {{
        if !$has_feature && !$span.allows_unstable($name) {
            #[allow(unused_mut)]
            let mut diag = feature_err($visitor.sess, $name, $span, $explain);
            $(for &note in $notes { diag.note(note); })?
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
                feature_err($visitor.sess, sym::$feature, spans, $explain).emit();
            }
        }
    }};
}

pub fn check_attribute(attr: &ast::Attribute, sess: &Session, features: &Features) {
    PostExpansionVisitor {
        sess,
        features,
        let_binding: None,
        handled_closure_lifetime_binders: FxHashSet::default(),
    }
    .visit_attribute(attr)
}

struct PostExpansionVisitor<'a> {
    sess: &'a Session,

    // `sess` contains a `Features`, but this might not be that one.
    features: &'a Features,

    /// Set while visiting the initializer of a `let` binding whose RHS is directly a closure.
    /// Used to suggest moving `for<...>` binders onto the binding's type.
    let_binding: Option<&'a ast::Local>,

    /// Binder spans for which we already emitted the `closure_lifetime_binder` gate while walking
    /// the live AST. Remaining pre-expansion spans (e.g. under `#[cfg(false)]`) are gated later.
    handled_closure_lifetime_binders: FxHashSet<Span>,
}

// -----------------------------------------------------------------------------
// POST-EXPANSION FEATURE GATES FOR UNSTABLE ATTRIBUTES ETC.
// **LEGACY**  POST-EXPANSION FEATURE GATES FOR UNSTABLE SYNTAX  **LEGACY**
// -----------------------------------------------------------------------------

// IMPORTANT: Don't add any new post-expansion feature gates for new unstable syntax!
//            It's a legacy mechanism for them.
//            Instead, register a pre-expansion feature gate using `gate_all` in fn `check_crate`.

impl<'a> PostExpansionVisitor<'a> {
    /// Gate `for<...>` binders on closures, suggesting a `fn` pointer binding type when possible.
    fn gate_closure_lifetime_binder(&mut self, closure: &ast::Closure, binder_span: Span) {
        self.handled_closure_lifetime_binders.insert(binder_span);

        if self.features.closure_lifetime_binder()
            || binder_span.allows_unstable(sym::closure_lifetime_binder)
        {
            return;
        }

        let mut err = feature_err(
            self.sess,
            sym::closure_lifetime_binder,
            binder_span,
            "`for<...>` binders for closures are experimental",
        );

        if let Some(sugg) =
            closure_lifetime_binder_binding_type_sugg(self.sess, self.let_binding, closure)
        {
            err.subdiagnostic(sugg);
        } else {
            err.help("consider removing `for<...>`");
        }

        err.emit();
    }

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
                            self.vis,
                            impl_trait_in_assoc_type,
                            ty.span,
                            "`impl Trait` in associated types is unstable"
                        );
                    } else {
                        gate!(
                            self.vis,
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
            msg!("only lifetime parameters can be used in this context")
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
                self.sess.dcx().emit_err(diagnostics::ForbiddenConstParam { const_param_spans });
            }
        }

        for param in params {
            if !param.bounds.is_empty() {
                let spans: Vec<_> = param.bounds.iter().map(|b| b.span()).collect();
                if param.bounds.iter().any(|bound| matches!(bound, GenericBound::Trait(_))) {
                    // Issue #149695
                    // Abort immediately otherwise items defined in complex bounds will be lowered into HIR,
                    // which will cause ICEs when errors of the items visit unlowered parents.
                    self.sess.dcx().emit_fatal(diagnostics::ForbiddenBound { spans });
                } else {
                    self.sess.dcx().emit_err(diagnostics::ForbiddenBound { spans });
                }
            }
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
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
                        auto_cfg => doc_cfg
                        masked => doc_masked
                        notable_trait => doc_notable_trait
                    }
                    "meant for internal use only" {
                        attribute => rustdoc_internals
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
            ast::ItemKind::Impl(ast::Impl { of_trait: Some(of_trait), .. }) => {
                if let ast::ImplPolarity::Negative(span) = of_trait.polarity {
                    gate!(
                        self,
                        negative_impls,
                        span.to(of_trait.trait_ref.path.span),
                        "negative impls are experimental",
                        "use marker types for now"
                    );
                }

                if let ast::Defaultness::Default(_) = of_trait.defaultness {
                    gate!(self, specialization, i.span, "specialization is experimental");
                }
            }

            ast::ItemKind::Trait(ast::Trait { is_auto: ast::IsAuto::Yes, .. }) => {
                gate!(self, auto_traits, i.span, "auto traits are experimental and possibly buggy");
            }

            ast::ItemKind::TraitAlias(..) => {
                gate!(self, trait_alias, i.span, "trait aliases are experimental");
            }

            ast::ItemKind::MacroDef(_, ast::MacroDef { macro_rules: false, .. }) => {
                let msg = "`macro` is experimental";
                gate!(self, decl_macro, i.span, msg);
            }

            ast::ItemKind::TyAlias(ast::TyAlias { ty: Some(ty), .. }) => {
                self.check_impl_trait(ty, false)
            }
            ast::ItemKind::Const(ast::ConstItem {
                kind: ast::ConstItemKind::TypeConst, ..
            }) => {
                // Make sure this is only allowed if the feature gate is enabled.
                // #![feature(min_generic_const_args)]
                gate!(self, min_generic_const_args, i.span, "top-level `type const` are unstable");
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
                        self,
                        link_llvm_intrinsics,
                        i.span,
                        "linking to LLVM intrinsics is experimental"
                    );
                }
            }
            ast::ForeignItemKind::TyAlias(..) => {
                gate!(self, extern_types, i.span, "extern types are experimental");
            }
            ast::ForeignItemKind::MacCall(..) => {}
        }

        visit::walk_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match &ty.kind {
            ast::TyKind::FnPtr(fn_ptr_ty) => {
                // Function pointers cannot be `const`
                self.check_late_bound_lifetime_defs(&fn_ptr_ty.generic_params);
            }
            ast::TyKind::Never => {
                gate!(self, never_type, ty.span, "the `!` type is experimental");
            }
            ast::TyKind::Pat(..) => {
                gate!(self, pattern_types, ty.span, "pattern types are unstable");
            }
            ast::TyKind::View(..) => {
                gate!(self, view_types, ty.span, "view types are unstable");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_where_predicate_kind(&mut self, kind: &'a ast::WherePredicateKind) {
        if let ast::WherePredicateKind::BoundPredicate(bound) = kind {
            // A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
            self.check_late_bound_lifetime_defs(&bound.bound_generic_params);
        }
        visit::walk_where_predicate_kind(self, kind);
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
            gate!(self, never_type, ty.span, "the `!` type is experimental");
        }
        visit::walk_generic_args(self, args);
    }

    fn visit_local(&mut self, local: &'a ast::Local) {
        // Only track direct `let pat = for<'a> |...| ...` inits; parenthesized or otherwise
        // wrapped closures fall back to the simpler help.
        if let Some(init) = local.kind.init()
            && matches!(init.kind, ast::ExprKind::Closure(_))
        {
            let prev = self.let_binding.replace(local);
            visit::walk_local(self, local);
            self.let_binding = prev;
        } else {
            visit::walk_local(self, local);
        }
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        match &e.kind {
            ast::ExprKind::Closure(closure) => {
                if let ast::ClosureBinder::For { span, .. } = &closure.binder {
                    self.gate_closure_lifetime_binder(closure, *span);
                }
                // Nested expressions inside the closure are not the `let` initializer.
                let prev = self.let_binding.take();
                visit::walk_expr(self, e);
                self.let_binding = prev;
                return;
            }
            ast::ExprKind::TryBlock(_, None) => {
                // `try { ... }` is old and is only gated post-expansion here.
                gate!(self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::TryBlock(_, Some(_)) => {
                // `try_blocks_heterogeneous` is new, and gated pre-expansion instead.
            }
            ast::ExprKind::Lit(token::Lit {
                kind: token::LitKind::Float | token::LitKind::Integer,
                suffix,
                ..
            }) => match *suffix {
                Some(sym::f16) => {
                    gate!(self, f16, e.span, "the type `f16` is unstable")
                }
                Some(sym::f128) => {
                    gate!(self, f128, e.span, "the type `f128` is unstable")
                }
                _ => {}
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
                            self,
                            half_open_range_patterns_in_slices,
                            pat.span,
                            "`X..` patterns in slices are experimental"
                        );
                    }
                }
            }
            PatKind::Box(..) => {
                gate!(self, box_patterns, pattern.span, "box pattern syntax is experimental");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef) {
        self.check_late_bound_lifetime_defs(&t.bound_generic_params);
        visit::walk_poly_trait_ref(self, t);
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, _: &AttrVec, _: Span, _: NodeId) {
        if let Some(_header) = fn_kind.header() {
            // Stability of const fn methods are covered in `visit_assoc_item` below.
        }

        if let FnKind::Closure(ast::ClosureBinder::For { generic_params, .. }, ..) = fn_kind {
            self.check_late_bound_lifetime_defs(generic_params);
        }

        visit::walk_fn(self, fn_kind)
    }

    fn visit_assoc_item(&mut self, i: &'a ast::AssocItem, ctxt: AssocCtxt) {
        let is_fn = match &i.kind {
            ast::AssocItemKind::Fn(_) => true,
            ast::AssocItemKind::Type(ast::TyAlias { ty, .. }) => {
                if let (Some(_), AssocCtxt::Trait) = (ty, ctxt) {
                    gate!(
                        self,
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
            ast::AssocItemKind::Const(ast::ConstItem {
                body,
                kind: ast::ConstItemKind::TypeConst,
                ..
            }) => {
                // Make sure this is only allowed if the feature gate is enabled.
                // #![feature(min_generic_const_args)]
                gate!(self, min_generic_const_args, i.span, "associated `type const` are unstable");
                // Make sure associated `type const` defaults in traits are only allowed
                // if the feature gate is enabled.
                // #![feature(associated_type_defaults)]
                if ctxt == AssocCtxt::Trait && body.is_some() {
                    gate!(
                        self,
                        associated_type_defaults,
                        i.span,
                        "associated type defaults are unstable"
                    );
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
                "specialization is experimental"
            );
        }
        visit::walk_assoc_item(self, i, ctxt)
    }
}

// -----------------------------------------------------------------------------

pub fn check_crate(krate: &ast::Crate, sess: &Session, features: &Features) {
    maybe_stage_features(sess, features, krate);
    check_incompatible_features(sess, features);
    check_dependent_features(sess, features);
    check_new_solver_banned_features(sess, features);
    check_features_requiring_new_solver(sess, features);

    let mut visitor = PostExpansionVisitor {
        sess,
        features,
        let_binding: None,
        handled_closure_lifetime_binders: FxHashSet::default(),
    };

    // -----------------------------------------------------------------------------
    // PRE-EXPANSION FEATURE GATES FOR UNSTABLE SYNTAX
    // -----------------------------------------------------------------------------

    let spans = sess.psess.gated_spans.spans.borrow();
    macro_rules! gate_all {
        ($feature:ident, $explain:literal $(, $help:literal)?) => {
            for &span in spans.get(&sym::$feature).into_flat_iter() {
                gate!(visitor, $feature, span, $explain $(, $help)?);
            }
        };
    }

    // tidy-alphabetical-start
    gate_all!(async_for_loop, "`for await` loops are experimental");
    gate_all!(builtin_syntax, "`builtin #` syntax is unstable");
    gate_all!(const_block_items, "const block items are experimental");
    gate_all!(const_closures, "const closures are experimental");
    gate_all!(const_trait_impl, "const trait impls are experimental");
    gate_all!(contracts, "contracts are incomplete");
    gate_all!(contracts_internals, "contract internal machinery is for internal use only");
    gate_all!(coroutines, "coroutine syntax is experimental");
    gate_all!(default_field_values, "default values on fields are experimental");
    gate_all!(ergonomic_clones, "ergonomic clones are experimental");
    gate_all!(explicit_tail_calls, "`become` expression is experimental");
    gate_all!(final_associated_functions, "`final` on trait functions is experimental");
    gate_all!(fn_delegation, "functions delegation is not yet fully implemented");
    gate_all!(frontmatter, "frontmatters are experimental");
    gate_all!(gen_blocks, "gen blocks are experimental");
    gate_all!(generic_const_items, "generic const items are experimental");
    gate_all!(global_registration, "global registration is experimental");
    gate_all!(guard_patterns, "guard patterns are experimental", "consider using match arm guards");
    gate_all!(impl_restriction, "`impl` restrictions are experimental");
    gate_all!(min_generic_const_args, "unbraced const blocks as const args are experimental");
    gate_all!(more_qualified_paths, "usage of qualified paths in this context is experimental");
    gate_all!(move_expr, "`move(expr)` syntax is experimental");
    gate_all!(mut_ref, "mutable by-reference bindings are experimental");
    gate_all!(mut_restriction, "`mut` restrictions are experimental");
    gate_all!(pin_ergonomics, "pinned reference syntax is experimental");
    gate_all!(postfix_match, "postfix match is experimental");
    gate_all!(return_type_notation, "return type notation is experimental");
    gate_all!(
        splat,
        "`fn(#[rustc_splat] (a, ...))` is incomplete",
        "call as func((a, ...)) instead"
    );
    gate_all!(super_let, "`super let` is experimental");
    gate_all!(try_blocks_heterogeneous, "`try bikeshed` expression is experimental");
    gate_all!(unnamed_enum_variants, "unnamed enum variants are experimental");
    gate_all!(unsafe_binders, "unsafe binder types are experimental");
    gate_all!(unsafe_fields, "`unsafe` fields are experimental");
    gate_all!(view_types, "view types are experimental");
    gate_all!(where_clause_attrs, "attributes in `where` clause are unstable");
    gate_all!(yeet_expr, "`do yeet` expression is experimental");
    // tidy-alphabetical-end

    gate_all!(
        async_trait_bounds,
        "`async` trait bounds are unstable",
        "use the desugared name of the async trait, such as `AsyncFn`"
    );
    // `closure_lifetime_binder` is gated in `PostExpansionVisitor` (with a richer suggestion when
    // possible). Spans not seen there — notably under `#[cfg(false)]` — are handled after the walk.
    gate_all!(
        half_open_range_patterns_in_slices,
        "half-open range patterns in slices are unstable"
    );
    gate_all!(
        named_fn_trait_parameters,
        "named parameters in parenthesized generic argument lists are experimental"
    );

    // `associated_const_equality` will be stabilized as part of `min_generic_const_args`.
    for &span in spans.get(&sym::associated_const_equality).into_flat_iter() {
        gate!(visitor, min_generic_const_args, span, "associated const equality is incomplete");
    }

    // `mgca_type_const_syntax` is part of `min_generic_const_args` so if
    // either or both are enabled we don't need to emit a feature error.
    for &span in spans.get(&sym::mgca_type_const_syntax).into_flat_iter() {
        if visitor.features.min_generic_const_args()
            || visitor.features.mgca_type_const_syntax()
            || span.allows_unstable(sym::min_generic_const_args)
            || span.allows_unstable(sym::mgca_type_const_syntax)
        {
            continue;
        }
        feature_err(
            visitor.sess,
            sym::min_generic_const_args,
            span,
            "`type const` syntax is experimental",
        )
        .emit();
    }

    // Negative bounds are *super* internal. We require `-Zinternal-testing-features` *and*
    // `#![feature(negative_bounds)]` to prevent proliferation. Under no circumstances do we
    // want to advertise the flag and the feature name to users!
    //
    // IMPORTANT: If you intend on turning negative bounds into a public-facing feature, please
    //            consult T-types and T-lang first! Do **not** just remove the `-Z` check!
    //
    // NOTE: `T: !Bound` means "`T` implements `Bound` negatively",
    //       it does **not** mean "`T` doesn't implement `Bound` (positively or negatively)"!
    //       The latter would be a SemVer hazard!
    if !sess.opts.unstable_opts.internal_testing_features || !visitor.features.negative_bounds() {
        for &span in spans.get(&sym::negative_bounds).into_flat_iter() {
            sess.dcx().emit_err(diagnostics::NegativeBoundUnsupported { span });
        }
    }

    if !visitor.features.never_patterns() {
        for &span in spans.get(&sym::never_patterns).into_flat_iter() {
            if span.allows_unstable(sym::never_patterns) {
                continue;
            }
            // We gate two types of spans: the span of a `!` pattern, and the span of a
            // match arm without a body. For the latter we want to give the user a normal
            // error.
            if let Ok("!") = sess.source_map().span_to_snippet(span).as_deref() {
                feature_err(sess, sym::never_patterns, span, "`!` patterns are experimental")
                    .emit();
            } else {
                let suggestion = span.shrink_to_hi();
                sess.dcx().emit_err(diagnostics::MatchArmWithNoBody { span, suggestion });
            }
        }
    }

    // Yield exprs can be enabled either by `yield_expr`, by `coroutines` or by `gen_blocks`.
    for &span in spans.get(&sym::yield_expr).into_flat_iter() {
        if (!visitor.features.coroutines() && !span.allows_unstable(sym::coroutines))
            && (!visitor.features.gen_blocks() && !span.allows_unstable(sym::gen_blocks))
            && (!visitor.features.yield_expr() && !span.allows_unstable(sym::yield_expr))
        {
            // Only mentioned `yield_expr` in the diagnostic since that'll be sufficient.
            // You can think of it as `coroutines` and `gen_blocks` implying `yield_expr`.
            feature_err(visitor.sess, sym::yield_expr, span, "yield syntax is experimental").emit();
        }
    }

    // -----------------------------------------------------------------------------
    // **LEGACY**  SOFT PRE-EXPANSION FEATURE GATES FOR UNSTABLE SYNTAX  **LEGACY**
    // -----------------------------------------------------------------------------

    // IMPORTANT: Do not extend the list below! New syntax should go above and use `gate_all`.

    // FIXME(#154045): Migrate all of these to erroring feature gates and
    //                 remove the corresponding post-expansion feature gates.

    macro_rules! soft_gate_all_legacy_dont_use {
        ($feature:ident, $explain:literal) => {
            for &span in spans.get(&sym::$feature).into_flat_iter() {
                if !visitor.features.$feature() && !span.allows_unstable(sym::$feature) {
                    feature_warn(&visitor.sess, sym::$feature, span, $explain);
                }
            }
        };
    }

    // tidy-alphabetical-start
    soft_gate_all_legacy_dont_use!(auto_traits, "`auto` traits are unstable");
    soft_gate_all_legacy_dont_use!(box_patterns, "box pattern syntax is experimental");
    soft_gate_all_legacy_dont_use!(decl_macro, "`macro` is experimental");
    soft_gate_all_legacy_dont_use!(negative_impls, "negative impls are experimental");
    soft_gate_all_legacy_dont_use!(specialization, "specialization is experimental");
    soft_gate_all_legacy_dont_use!(trait_alias, "trait aliases are experimental");
    soft_gate_all_legacy_dont_use!(try_blocks, "`try` blocks are unstable");
    // tidy-alphabetical-end

    for &span in spans.get(&sym::min_specialization).into_flat_iter() {
        if !visitor.features.specialization()
            && !visitor.features.min_specialization()
            && !span.allows_unstable(sym::specialization)
            && !span.allows_unstable(sym::min_specialization)
        {
            feature_warn(visitor.sess, sym::specialization, span, "specialization is experimental");
        }
    }

    // -----------------------------------------------------------------------------

    visit::walk_crate(&mut visitor, krate);

    // Reject `for<...>` closure binders that never reached the AST walk (e.g. `#[cfg(false)]`).
    if !visitor.features.closure_lifetime_binder() {
        for &span in spans.get(&sym::closure_lifetime_binder).into_flat_iter() {
            if span.allows_unstable(sym::closure_lifetime_binder)
                || visitor.handled_closure_lifetime_binders.contains(&span)
            {
                continue;
            }
            feature_err(
                sess,
                sym::closure_lifetime_binder,
                span,
                "`for<...>` binders for closures are experimental",
            )
            .with_help("consider removing `for<...>`")
            .emit();
        }
    }
}

/// Build a suggestion rewriting
/// `let cl = for<'a> |x: &'a T| -> U { ... }` into
/// `let cl: for<'a> fn(&'a T) -> U = |x| { ... }` when that is a reasonable alternative.
fn closure_lifetime_binder_binding_type_sugg(
    sess: &Session,
    local: Option<&ast::Local>,
    closure: &ast::Closure,
) -> Option<diagnostics::ClosureLifetimeBinderBindingTypeSugg> {
    let local = local?;
    if local.ty.is_some() {
        return None;
    }
    // Only by-value `let ident = ...` / `let mut ident = ...` bindings.
    if !matches!(&local.pat.kind, PatKind::Ident(BindingMode(ByRef::No, _), _, None)) {
        return None;
    }

    // Explicit `move`/`use`/`async`/`const`/`static` closures are not `fn` pointers.
    if !matches!(closure.capture_clause, ast::CaptureBy::Ref)
        || closure.coroutine_kind.is_some()
        || matches!(closure.constness, ast::Const::Yes(_))
        || matches!(closure.movability, ast::Movability::Static)
    {
        return None;
    }

    let ast::ClosureBinder::For { span: binder_span, generic_params } = &closure.binder else {
        return None;
    };

    // `for<T>` / `for<'a: 'static>` are not valid on `fn` pointer types.
    if !generic_params
        .iter()
        .all(|param| matches!(param.kind, GenericParamKind::Lifetime) && param.bounds.is_empty())
    {
        return None;
    }

    // Need fully explicit parameter and return types to form a useful `fn` type. A top-level or
    // nested `_` (e.g. `-> _`, `&'a _`) must not be copied into a MachineApplicable suggestion.
    let ast::FnRetTy::Ty(ret_ty) = &closure.fn_decl.output else {
        return None;
    };
    if ty_contains_infer(ret_ty)
        || closure.fn_decl.inputs.iter().any(|param| ty_contains_infer(&param.ty))
    {
        return None;
    }

    // Only by-value binding patterns (and `_`) can be rewritten safely.
    if !closure.fn_decl.inputs.iter().all(|param| {
        matches!(
            &param.pat.kind,
            PatKind::Wild | PatKind::Ident(BindingMode(ByRef::No, _), _, None)
        )
    }) {
        return None;
    }

    // `pprust::pat_to_string` drops parameter attributes; don't emit a lossy rewrite.
    if closure.fn_decl.inputs.iter().any(|param| !param.attrs.is_empty()) {
        return None;
    }

    // Don't rewrite macro-expanded closures; hygiene makes capture analysis unreliable and the
    // suggestion would point into the macro definition.
    if binder_span.from_expansion() || closure.fn_decl_span.from_expansion() {
        return None;
    }

    let binder = sess.source_map().span_to_snippet(*binder_span).ok()?;
    let inputs: String = closure
        .fn_decl
        .inputs
        .iter()
        .map(|param| pprust::ty_to_string(&param.ty))
        .intersperse(", ".to_string())
        .collect();
    let ty = format!("{binder} fn({inputs}) -> {}", pprust::ty_to_string(ret_ty));

    let closure_pats: String = closure
        .fn_decl
        .inputs
        .iter()
        .map(|param| pprust::pat_to_string(&param.pat))
        .intersperse(", ".to_string())
        .collect();

    let binding = local.pat.span.shrink_to_hi();
    let closure_header = binder_span.to(closure.fn_decl_span);
    let closure_code = format!("|{closure_pats}|");

    // `CaptureBy::Ref` only means no `move`/`use`. Without name resolution, any other simple
    // path may be an env capture (including uppercase locals) or a free item. Offer the rewrite
    // only as maybe-incorrect in that case so rustfix won't auto-apply a breaking change.
    // Paths bound locally in the body (e.g. `let n = ...; n`) are fine for `fn` pointers.
    if closure_body_has_free_simple_path(closure) {
        Some(diagnostics::ClosureLifetimeBinderBindingTypeSugg::MaybeIncorrect {
            binding,
            ty,
            closure_header,
            closure: closure_code,
        })
    } else {
        Some(diagnostics::ClosureLifetimeBinderBindingTypeSugg::MachineApplicable {
            binding,
            ty,
            closure_header,
            closure: closure_code,
        })
    }
}

/// Returns true if `ty` contains any `_` inference placeholder, including nested forms like
/// `&'a _` or `(_, u8)`.
fn ty_contains_infer(ty: &ast::Ty) -> bool {
    struct InferVisitor {
        found: bool,
    }

    impl<'a> Visitor<'a> for InferVisitor {
        fn visit_ty(&mut self, ty: &'a ast::Ty) {
            if self.found {
                return;
            }
            if matches!(ty.kind, ast::TyKind::Infer) {
                self.found = true;
                return;
            }
            visit::walk_ty(self, ty);
        }
    }

    let mut visitor = InferVisitor { found: false };
    visitor.visit_ty(ty);
    visitor.found
}

/// Returns true if the closure body contains a single-segment value path that is neither a
/// parameter nor a name bound inside the body.
///
/// Locals are tracked as hygiene-aware [`Ident`]s (name + `SyntaxContext`) so a macro parameter
/// `$x` is not confused with a closure parameter `x` that happens to share a spelling.
///
/// This is intentionally AST-only and conservative: free functions and constructors look the same
/// as captures here. Callers should downgrade suggestion applicability when this is true.
fn closure_body_has_free_simple_path(closure: &ast::Closure) -> bool {
    let mut known_locals = FxHashSet::default();
    for param in &closure.fn_decl.inputs {
        if let PatKind::Ident(_, ident, _) = param.pat.kind {
            known_locals.insert(ident);
        }
    }

    struct FreePathVisitor {
        known_locals: FxHashSet<Ident>,
        has_free_path: bool,
    }

    impl FreePathVisitor {
        fn bind_pat(&mut self, pat: &ast::Pat) {
            match &pat.kind {
                PatKind::Ident(_, ident, sub) => {
                    self.known_locals.insert(*ident);
                    if let Some(sub) = sub {
                        self.bind_pat(sub);
                    }
                }
                PatKind::Tuple(pats)
                | PatKind::TupleStruct(_, _, pats)
                | PatKind::Slice(pats)
                | PatKind::Or(pats) => {
                    for pat in pats {
                        self.bind_pat(pat);
                    }
                }
                PatKind::Struct(_, _, fields, _) => {
                    for field in fields {
                        self.bind_pat(&field.pat);
                    }
                }
                PatKind::Box(pat)
                | PatKind::Deref(pat)
                | PatKind::Ref(pat, ..)
                | PatKind::Paren(pat) => self.bind_pat(pat),
                _ => {}
            }
        }
    }

    impl<'a> Visitor<'a> for FreePathVisitor {
        fn visit_ty(&mut self, _: &'a ast::Ty) {
            // Paths in types are not value captures.
        }

        fn visit_block(&mut self, block: &'a ast::Block) {
            let old = self.known_locals.clone();
            visit::walk_block(self, block);
            self.known_locals = old;
        }

        fn visit_local(&mut self, local: &'a ast::Local) {
            // Visit the initializer (and `else` block) before binding names from the pattern.
            // Bindings are not in scope in the `else` block.
            if let Some((init, els)) = local.kind.init_else_opt() {
                self.visit_expr(init);
                if let Some(els) = els {
                    // Must go through `visit_block` so locals declared in the `else` block do not
                    // leak into `known_locals` for code after the `let else`.
                    self.visit_block(els);
                }
            }
            self.bind_pat(&local.pat);
        }

        fn visit_arm(&mut self, arm: &'a ast::Arm) {
            let old = self.known_locals.clone();
            self.bind_pat(&arm.pat);
            visit::walk_arm(self, arm);
            self.known_locals = old;
        }

        fn visit_expr(&mut self, expr: &'a ast::Expr) {
            if self.has_free_path {
                return;
            }
            if let ast::ExprKind::Path(None, path) = &expr.kind
                && let [seg] = path.segments.as_slice()
                && seg.args.is_none()
                && !self.known_locals.contains(&seg.ident)
            {
                self.has_free_path = true;
                return;
            }
            match &expr.kind {
                // `let` bindings from let-chains / `if let` / `while let` conditions. The enclosing
                // `If` / `While` arms restore `known_locals` so these do not escape that scope.
                ast::ExprKind::Let(pat, scrutinee, _, _) => {
                    self.visit_expr(scrutinee);
                    self.bind_pat(pat);
                }
                // `if`/`if let`/`if` let-chains: condition bindings are in scope for the then
                // branch only, not the else branch or anything after the `if`.
                ast::ExprKind::If(cond, then_block, else_opt) => {
                    let old = self.known_locals.clone();
                    self.visit_expr(cond);
                    self.visit_block(then_block);
                    self.known_locals = old;
                    if let Some(els) = else_opt {
                        self.visit_expr(els);
                    }
                }
                // `while`/`while let`: condition bindings are in scope for the loop body only.
                ast::ExprKind::While(cond, body, _) => {
                    let old = self.known_locals.clone();
                    self.visit_expr(cond);
                    self.visit_block(body);
                    self.known_locals = old;
                }
                ast::ExprKind::ForLoop(for_loop) => {
                    self.visit_expr(&for_loop.iter);
                    let old = self.known_locals.clone();
                    self.bind_pat(&for_loop.pat);
                    self.visit_block(&for_loop.body);
                    self.known_locals = old;
                }
                _ => visit::walk_expr(self, expr),
            }
        }
    }

    let mut visitor = FreePathVisitor { known_locals, has_free_path: false };
    visitor.visit_expr(&closure.body);
    visitor.has_free_path
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

    if let Some(Attribute::Parsed(AttributeKind::Feature(feature_idents, first_span))) =
        AttributeParser::parse_limited_sym(sess, &krate.attrs, &[sym::feature])
    {
        // `feature(...)` used on non-nightly. This is definitely an error.
        let mut err = diagnostics::FeatureOnNonNightly {
            span: first_span,
            channel: option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)"),
            stable_features: vec![],
            sugg: None,
        };

        let mut all_stable = true;
        for ident in feature_idents {
            let name = ident.name;
            let stable_since = features
                .enabled_lang_features()
                .iter()
                .find(|feat| feat.gate_name == name)
                .map(|feat| feat.stable_since)
                .flatten();
            if let Some(since) = stable_since {
                err.stable_features.push(diagnostics::StableFeature { name, since });
            } else {
                all_stable = false;
            }
        }
        if all_stable {
            err.sugg = Some(first_span);
        }
        sess.dcx().emit_err(err);
        errored = true;
    }
    // Just make sure we actually error if anything is listed in `enabled_features`.
    assert!(errored);
}

fn check_incompatible_features(sess: &Session, features: &Features) {
    let enabled_features = features.enabled_features_iter_stable_order();

    for (f1, f2) in rustc_feature::INCOMPATIBLE_FEATURES
        .iter()
        .filter(|(f1, f2)| features.enabled(*f1) && features.enabled(*f2))
    {
        if let Some((f1_name, f1_span)) = enabled_features.clone().find(|(name, _)| name == f1)
            && let Some((f2_name, f2_span)) = enabled_features.clone().find(|(name, _)| name == f2)
        {
            let spans = vec![f1_span, f2_span];
            sess.dcx().emit_err(diagnostics::IncompatibleFeatures {
                spans,
                f1: f1_name,
                f2: f2_name,
            });
        }
    }
}

fn check_dependent_features(sess: &Session, features: &Features) {
    for &(parent, children) in
        rustc_feature::DEPENDENT_FEATURES.iter().filter(|(parent, _)| features.enabled(*parent))
    {
        if children.iter().any(|f| !features.enabled(*f)) {
            let parent_span = features
                .enabled_features_iter_stable_order()
                .find_map(|(name, span)| (name == parent).then_some(span))
                .unwrap();
            // FIXME: should probably format this in fluent instead of here
            let missing = children
                .iter()
                .filter(|f| !features.enabled(**f))
                .map(|s| format!("`{}`", s.as_str()))
                .intersperse(String::from(", "))
                .collect();
            sess.dcx().emit_err(diagnostics::MissingDependentFeatures {
                parent_span,
                parent,
                missing,
            });
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
        // Abort immediately, otherwise GCE can lower to `ConstKind::Expr`,
        // which the new solver intentionally does not support.
        #[allow(rustc::symbol_intern_string_literal)]
        sess.dcx().emit_fatal(diagnostics::IncompatibleFeatures {
            spans: vec![gce_span],
            f1: Symbol::intern("-Znext-solver=globally"),
            f2: sym::generic_const_exprs,
        });
    }
}

fn check_features_requiring_new_solver(sess: &Session, features: &Features) {
    if sess.opts.unstable_opts.next_solver.globally {
        return;
    }

    // Require the new solver with GCA, because the old solver can't implement GCA correctly as it
    // does not support normalization obligations for free and inherent consts.
    if let Some(gca_span) = features
        .enabled_lang_features()
        .iter()
        .find(|feat| feat.gate_name == sym::generic_const_args)
        .map(|feat| feat.attr_sp)
    {
        #[allow(rustc::symbol_intern_string_literal)]
        sess.dcx().emit_err(diagnostics::MissingDependentFeatures {
            parent_span: gca_span,
            parent: sym::generic_const_args,
            missing: String::from("-Znext-solver=globally"),
        });
    }
}
