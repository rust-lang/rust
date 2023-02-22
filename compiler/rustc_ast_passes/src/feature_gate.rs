use rustc_ast as ast;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::{AssocConstraint, AssocConstraintKind, NodeId};
use rustc_ast::{PatKind, RangeEnd};
use rustc_errors::{struct_span_err, Applicability, StashKey};
use rustc_feature::{AttributeGate, BuiltinAttribute, Features, GateIssue, BUILTIN_ATTRIBUTE_MAP};
use rustc_session::parse::{feature_err, feature_err_issue, feature_warn};
use rustc_session::Session;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::spec::abi;
use thin_vec::ThinVec;
use tracing::debug;

use crate::errors::ForbiddenLifetimeBound;

macro_rules! gate_feature_fn {
    ($visitor: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $help: expr) => {{
        let (visitor, has_feature, span, name, explain, help) =
            (&*$visitor, $has_feature, $span, $name, $explain, $help);
        let has_feature: bool = has_feature(visitor.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            feature_err(&visitor.sess.parse_sess, name, span, explain).help(help).emit();
        }
    }};
    ($visitor: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr) => {{
        let (visitor, has_feature, span, name, explain) =
            (&*$visitor, $has_feature, $span, $name, $explain);
        let has_feature: bool = has_feature(visitor.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            feature_err(&visitor.sess.parse_sess, name, span, explain).emit();
        }
    }};
    (future_incompatible; $visitor: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr) => {{
        let (visitor, has_feature, span, name, explain) =
            (&*$visitor, $has_feature, $span, $name, $explain);
        let has_feature: bool = has_feature(visitor.features);
        debug!(
            "gate_feature(feature = {:?}, span = {:?}); has? {} (future_incompatible)",
            name, span, has_feature
        );
        if !has_feature && !span.allows_unstable($name) {
            feature_warn(&visitor.sess.parse_sess, name, span, explain);
        }
    }};
}

macro_rules! gate_feature_post {
    ($visitor: expr, $feature: ident, $span: expr, $explain: expr, $help: expr) => {
        gate_feature_fn!($visitor, |x: &Features| x.$feature, $span, sym::$feature, $explain, $help)
    };
    ($visitor: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($visitor, |x: &Features| x.$feature, $span, sym::$feature, $explain)
    };
    (future_incompatible; $visitor: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!(future_incompatible; $visitor, |x: &Features| x.$feature, $span, sym::$feature, $explain)
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
    fn check_abi(&self, abi: ast::StrLit, constness: ast::Const) {
        let ast::StrLit { symbol_unescaped, span, .. } = abi;

        if let ast::Const::Yes(_) = constness {
            match symbol_unescaped {
                // Stable
                sym::Rust | sym::C => {}
                abi => gate_feature_post!(
                    &self,
                    const_extern_fn,
                    span,
                    &format!("`{}` as a `const fn` ABI is unstable", abi)
                ),
            }
        }

        match abi::is_enabled(&self.features, span, symbol_unescaped.as_str()) {
            Ok(()) => (),
            Err(abi::AbiDisabled::Unstable { feature, explain }) => {
                feature_err_issue(
                    &self.sess.parse_sess,
                    feature,
                    span,
                    GateIssue::Language,
                    explain,
                )
                .emit();
            }
            Err(abi::AbiDisabled::Unrecognized) => {
                if self.sess.opts.pretty.map_or(true, |ppm| ppm.needs_hir()) {
                    self.sess.parse_sess.span_diagnostic.delay_span_bug(
                        span,
                        &format!(
                            "unrecognized ABI not caught in lowering: {}",
                            symbol_unescaped.as_str()
                        ),
                    );
                }
            }
        }
    }

    fn check_extern(&self, ext: ast::Extern, constness: ast::Const) {
        if let ast::Extern::Explicit(abi, _) = ext {
            self.check_abi(abi, constness);
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

    fn check_late_bound_lifetime_defs(&self, params: &[ast::GenericParam]) {
        // Check only lifetime parameters are present and that the lifetime
        // parameters that are present have no bounds.
        let non_lt_param_spans: Vec<_> = params
            .iter()
            .filter_map(|param| match param.kind {
                ast::GenericParamKind::Lifetime { .. } => None,
                _ => Some(param.ident.span),
            })
            .collect();
        // FIXME: gate_feature_post doesn't really handle multispans...
        if !non_lt_param_spans.is_empty() && !self.features.non_lifetime_binders {
            feature_err(
                &self.sess.parse_sess,
                sym::non_lifetime_binders,
                non_lt_param_spans,
                crate::fluent_generated::ast_passes_forbidden_non_lifetime_param,
            )
            .emit();
        }
        for param in params {
            if !param.bounds.is_empty() {
                let spans: Vec<_> = param.bounds.iter().map(|b| b.span()).collect();
                self.sess.emit_err(ForbiddenLifetimeBound { spans });
            }
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        let attr_info = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));
        // Check feature gates for built-in attributes.
        if let Some(BuiltinAttribute {
            gate: AttributeGate::Gated(_, name, descr, has_feature),
            ..
        }) = attr_info
        {
            gate_feature_fn!(self, has_feature, attr.span, *name, *descr);
        }
        // Check unstable flavors of the `#[doc]` attribute.
        if attr.has_name(sym::doc) {
            for nested_meta in attr.meta_item_list().unwrap_or_default() {
                macro_rules! gate_doc { ($($name:ident => $feature:ident)*) => {
                    $(if nested_meta.has_name(sym::$name) {
                        let msg = concat!("`#[doc(", stringify!($name), ")]` is experimental");
                        gate_feature_post!(self, $feature, attr.span, msg);
                    })*
                }}

                gate_doc!(
                    cfg => doc_cfg
                    cfg_hide => doc_cfg_hide
                    masked => doc_masked
                    notable_trait => doc_notable_trait
                );

                if nested_meta.has_name(sym::keyword) {
                    let msg = "`#[doc(keyword)]` is meant for internal use only";
                    gate_feature_post!(self, rustdoc_internals, attr.span, msg);
                }

                if nested_meta.has_name(sym::fake_variadic) {
                    let msg = "`#[doc(fake_variadic)]` is meant for internal use only";
                    gate_feature_post!(self, rustdoc_internals, attr.span, msg);
                }
            }
        }

        // Emit errors for non-staged-api crates.
        if !self.features.staged_api {
            if attr.has_name(sym::unstable)
                || attr.has_name(sym::stable)
                || attr.has_name(sym::rustc_const_unstable)
                || attr.has_name(sym::rustc_const_stable)
                || attr.has_name(sym::rustc_default_body_unstable)
            {
                struct_span_err!(
                    self.sess,
                    attr.span,
                    E0734,
                    "stability attributes may not be used outside of the standard library",
                )
                .emit();
            }
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match &i.kind {
            ast::ItemKind::ForeignMod(foreign_module) => {
                if let Some(abi) = foreign_module.abi {
                    self.check_abi(abi, ast::Const::No);
                }
            }

            ast::ItemKind::Fn(..) => {
                if self.sess.contains_name(&i.attrs, sym::start) {
                    gate_feature_post!(
                        &self,
                        start,
                        i.span,
                        "`#[start]` functions are experimental \
                         and their signature may change \
                         over time"
                    );
                }
            }

            ast::ItemKind::Struct(..) => {
                for attr in self.sess.filter_by_name(&i.attrs, sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
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

            ast::ItemKind::Impl(box ast::Impl { polarity, defaultness, of_trait, .. }) => {
                if let &ast::ImplPolarity::Negative(span) = polarity {
                    gate_feature_post!(
                        &self,
                        negative_impls,
                        span.to(of_trait.as_ref().map_or(span, |t| t.path.span)),
                        "negative trait bounds are not yet fully implemented; \
                         use marker types for now"
                    );
                }

                if let ast::Defaultness::Default(_) = defaultness {
                    gate_feature_post!(&self, specialization, i.span, "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(box ast::Trait { is_auto: ast::IsAuto::Yes, .. }) => {
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

            ast::ItemKind::TyAlias(box ast::TyAlias { ty: Some(ty), .. }) => {
                self.check_impl_trait(&ty)
            }

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
        match &ty.kind {
            ast::TyKind::BareFn(bare_fn_ty) => {
                // Function pointers cannot be `const`
                self.check_extern(bare_fn_ty.ext, ast::Const::No);
                self.check_late_bound_lifetime_defs(&bare_fn_ty.generic_params);
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span, "the `!` type is experimental");
            }
            ast::TyKind::TraitObject(_, ast::TraitObjectSyntax::DynStar, ..) => {
                gate_feature_post!(&self, dyn_star, ty.span, "dyn* trait objects are unstable");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        for predicate in &g.where_clause.predicates {
            match predicate {
                ast::WherePredicate::BoundPredicate(bound_pred) => {
                    // A type binding, eg `for<'c> Foo: Send+Clone+'c`
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

    fn visit_stmt(&mut self, stmt: &'a ast::Stmt) {
        if let ast::StmtKind::Semi(expr) = &stmt.kind
            && let ast::ExprKind::Assign(lhs, _, _) = &expr.kind
            && let ast::ExprKind::Type(..) = lhs.kind
            && self.sess.parse_sess.span_diagnostic.err_count() == 0
            && !self.features.type_ascription
            && !lhs.span.allows_unstable(sym::type_ascription)
        {
            // When we encounter a statement of the form `foo: Ty = val;`, this will emit a type
            // ascription error, but the likely intention was to write a `let` statement. (#78907).
            feature_err(
                &self.sess.parse_sess,
                sym::type_ascription,
                lhs.span,
                "type ascription is experimental",
            ).span_suggestion_verbose(
                lhs.span.shrink_to_lo(),
                "you might have meant to introduce a new binding",
                "let ".to_string(),
                Applicability::MachineApplicable,
            ).emit();
        }
        visit::walk_stmt(self, stmt);
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
                if self.sess.parse_sess.span_diagnostic.err_count() == 0 {
                    // To avoid noise about type ascription in common syntax errors,
                    // only emit if it is the *only* error.
                    gate_feature_post!(
                        &self,
                        type_ascription,
                        e.span,
                        "type ascription is experimental"
                    );
                } else {
                    // And if it isn't, cancel the early-pass warning.
                    self.sess
                        .parse_sess
                        .span_diagnostic
                        .steal_diagnostic(e.span, StashKey::EarlySyntaxWarning)
                        .map(|err| err.cancel());
                }
            }
            ast::ExprKind::TryBlock(_) => {
                gate_feature_post!(&self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::Closure(box ast::Closure { constness: ast::Const::Yes(_), .. }) => {
                gate_feature_post!(
                    &self,
                    const_closures,
                    e.span,
                    "const closures are experimental"
                );
            }
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
                        gate_feature_post!(
                            &self,
                            half_open_range_patterns_in_slices,
                            pat.span,
                            "`X..` patterns in slices are experimental"
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
            PatKind::Range(_, Some(_), Spanned { node: RangeEnd::Excluded, .. }) => {
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

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef) {
        self.check_late_bound_lifetime_defs(&t.bound_generic_params);
        visit::walk_poly_trait_ref(self, t);
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, span: Span, _: NodeId) {
        if let Some(header) = fn_kind.header() {
            // Stability of const fn methods are covered in `visit_assoc_item` below.
            self.check_extern(header.ext, header.constness);
        }

        if let FnKind::Closure(ast::ClosureBinder::For { generic_params, .. }, ..) = fn_kind {
            self.check_late_bound_lifetime_defs(generic_params);
        }

        if fn_kind.ctxt() != Some(FnCtxt::Foreign) && fn_kind.decl().c_variadic() {
            gate_feature_post!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind)
    }

    fn visit_assoc_constraint(&mut self, constraint: &'a AssocConstraint) {
        if let AssocConstraintKind::Bound { .. } = constraint.kind {
            gate_feature_post!(
                &self,
                associated_type_bounds,
                constraint.span,
                "associated type bounds are unstable"
            )
        }
        visit::walk_assoc_constraint(self, constraint)
    }

    fn visit_assoc_item(&mut self, i: &'a ast::AssocItem, ctxt: AssocCtxt) {
        let is_fn = match &i.kind {
            ast::AssocItemKind::Fn(_) => true,
            ast::AssocItemKind::Type(box ast::TyAlias { ty, .. }) => {
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
}

pub fn check_crate(krate: &ast::Crate, sess: &Session) {
    maybe_stage_features(sess, krate);
    check_incompatible_features(sess);
    let mut visitor = PostExpansionVisitor { sess, features: &sess.features_untracked() };

    let spans = sess.parse_sess.gated_spans.spans.borrow();
    macro_rules! gate_all {
        ($gate:ident, $msg:literal, $help:literal) => {
            if let Some(spans) = spans.get(&sym::$gate) {
                for span in spans {
                    gate_feature_post!(&visitor, $gate, *span, $msg, $help);
                }
            }
        };
        ($gate:ident, $msg:literal) => {
            if let Some(spans) = spans.get(&sym::$gate) {
                for span in spans {
                    gate_feature_post!(&visitor, $gate, *span, $msg);
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
        async_closure,
        "async closures are unstable",
        "to use an async block, remove the `||`: `async {`"
    );
    gate_all!(
        closure_lifetime_binder,
        "`for<...>` binders for closures are experimental",
        "consider removing `for<...>`"
    );
    gate_all!(more_qualified_paths, "usage of qualified paths in this context is experimental");
    gate_all!(generators, "yield syntax is experimental");
    gate_all!(raw_ref_op, "raw address of syntax is experimental");
    gate_all!(const_trait_impl, "const trait impls are experimental");
    gate_all!(
        half_open_range_patterns_in_slices,
        "half-open range patterns in slices are unstable"
    );
    gate_all!(inline_const, "inline-const is experimental");
    gate_all!(inline_const_pat, "inline-const in pattern position is experimental");
    gate_all!(associated_const_equality, "associated const equality is incomplete");
    gate_all!(yeet_expr, "`do yeet` expression is experimental");

    // All uses of `gate_all!` below this point were added in #65742,
    // and subsequently disabled (with the non-early gating readded).
    // We emit an early future-incompatible warning for these.
    // New syntax gates should go above here to get a hard error gate.
    macro_rules! gate_all {
        ($gate:ident, $msg:literal) => {
            for span in spans.get(&sym::$gate).unwrap_or(&vec![]) {
                gate_feature_post!(future_incompatible; &visitor, $gate, *span, $msg);
            }
        };
    }

    gate_all!(trait_alias, "trait aliases are experimental");
    gate_all!(associated_type_bounds, "associated type bounds are unstable");
    gate_all!(decl_macro, "`macro` is experimental");
    gate_all!(box_patterns, "box pattern syntax is experimental");
    gate_all!(exclusive_range_pattern, "exclusive range pattern syntax is experimental");
    gate_all!(try_blocks, "`try` blocks are unstable");
    gate_all!(box_syntax, "box expression syntax is experimental; you can call `Box::new` instead");
    gate_all!(type_ascription, "type ascription is experimental");

    visit::walk_crate(&mut visitor, krate);
}

fn maybe_stage_features(sess: &Session, krate: &ast::Crate) {
    // checks if `#![feature]` has been used to enable any lang feature
    // does not check the same for lib features unless there's at least one
    // declared lang feature
    if !sess.opts.unstable_features.is_nightly_build() {
        let lang_features = &sess.features_untracked().declared_lang_features;
        if lang_features.len() == 0 {
            return;
        }
        for attr in krate.attrs.iter().filter(|attr| attr.has_name(sym::feature)) {
            let mut err = struct_span_err!(
                sess.parse_sess.span_diagnostic,
                attr.span,
                E0554,
                "`#![feature]` may not be used on the {} release channel",
                option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)")
            );
            let mut all_stable = true;
            for ident in
                attr.meta_item_list().into_iter().flatten().flat_map(|nested| nested.ident())
            {
                let name = ident.name;
                let stable_since = lang_features
                    .iter()
                    .flat_map(|&(feature, _, since)| if feature == name { since } else { None })
                    .next();
                if let Some(since) = stable_since {
                    err.help(&format!(
                        "the feature `{}` has been stable since {} and no longer requires \
                                  an attribute to enable",
                        name, since
                    ));
                } else {
                    all_stable = false;
                }
            }
            if all_stable {
                err.span_suggestion(
                    attr.span,
                    "remove the attribute",
                    "",
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
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
                    spans,
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
