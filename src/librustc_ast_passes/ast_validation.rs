// Validate AST before lowering it to HIR.
//
// This pass is supposed to catch things that fit into AST data structures,
// but not permitted by the language. It runs after expansion when AST is frozen,
// so it can check for erroneous constructions produced by syntax extensions.
// This pass is supposed to perform only simple checks not requiring name resolution
// or type checking or some other kind of complex analysis.

use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{error_code, struct_span_err, Applicability, FatalError};
use rustc_parse::validate_attr;
use rustc_session::lint::builtin::PATTERNS_IN_FNS_WITHOUT_BODY;
use rustc_session::lint::LintBuffer;
use rustc_session::Session;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym};
use rustc_span::Span;
use std::mem;
use syntax::ast::*;
use syntax::attr;
use syntax::expand::is_proc_macro_attr;
use syntax::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use syntax::walk_list;

/// Is `self` allowed semantically as the first parameter in an `FnDecl`?
enum SelfSemantic {
    Yes,
    No,
}

/// A syntactic context that disallows certain kinds of bounds (e.g., `?Trait` or `?const Trait`).
#[derive(Clone, Copy)]
enum BoundContext {
    ImplTrait,
    TraitBounds,
    TraitObject,
}

impl BoundContext {
    fn description(&self) -> &'static str {
        match self {
            Self::ImplTrait => "`impl Trait`",
            Self::TraitBounds => "supertraits",
            Self::TraitObject => "trait objects",
        }
    }
}

struct AstValidator<'a> {
    session: &'a Session,

    /// The span of the `extern` in an `extern { ... }` block, if any.
    extern_mod: Option<&'a Item>,

    /// Are we inside a trait impl?
    in_trait_impl: bool,

    has_proc_macro_decls: bool,

    /// Used to ban nested `impl Trait`, e.g., `impl Into<impl Debug>`.
    /// Nested `impl Trait` _is_ allowed in associated type position,
    /// e.g., `impl Iterator<Item = impl Debug>`.
    outer_impl_trait: Option<Span>,

    /// Keeps track of the `BoundContext` as we recurse.
    ///
    /// This is used to forbid `?const Trait` bounds in, e.g.,
    /// `impl Iterator<Item = Box<dyn ?const Trait>`.
    bound_context: Option<BoundContext>,

    /// Used to ban `impl Trait` in path projections like `<impl Iterator>::Item`
    /// or `Foo::Bar<impl Trait>`
    is_impl_trait_banned: bool,

    /// Used to ban associated type bounds (i.e., `Type<AssocType: Bounds>`) in
    /// certain positions.
    is_assoc_ty_bound_banned: bool,

    lint_buffer: &'a mut LintBuffer,
}

impl<'a> AstValidator<'a> {
    fn with_in_trait_impl(&mut self, is_in: bool, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.in_trait_impl, is_in);
        f(self);
        self.in_trait_impl = old;
    }

    fn with_banned_impl_trait(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_impl_trait_banned, true);
        f(self);
        self.is_impl_trait_banned = old;
    }

    fn with_banned_assoc_ty_bound(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_assoc_ty_bound_banned, true);
        f(self);
        self.is_assoc_ty_bound_banned = old;
    }

    fn with_impl_trait(&mut self, outer: Option<Span>, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.outer_impl_trait, outer);
        if outer.is_some() {
            self.with_bound_context(BoundContext::ImplTrait, |this| f(this));
        } else {
            f(self)
        }
        self.outer_impl_trait = old;
    }

    fn with_bound_context(&mut self, ctx: BoundContext, f: impl FnOnce(&mut Self)) {
        let old = self.bound_context.replace(ctx);
        f(self);
        self.bound_context = old;
    }

    fn visit_assoc_ty_constraint_from_generic_args(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Equality { .. } => {}
            AssocTyConstraintKind::Bound { .. } => {
                if self.is_assoc_ty_bound_banned {
                    self.err_handler().span_err(
                        constraint.span,
                        "associated type bounds are not allowed within structs, enums, or unions",
                    );
                }
            }
        }
        self.visit_assoc_ty_constraint(constraint);
    }

    // Mirrors `visit::walk_ty`, but tracks relevant state.
    fn walk_ty(&mut self, t: &'a Ty) {
        match t.kind {
            TyKind::ImplTrait(..) => {
                self.with_impl_trait(Some(t.span), |this| visit::walk_ty(this, t))
            }
            TyKind::TraitObject(..) => {
                self.with_bound_context(BoundContext::TraitObject, |this| visit::walk_ty(this, t));
            }
            TyKind::Path(ref qself, ref path) => {
                // We allow these:
                //  - `Option<impl Trait>`
                //  - `option::Option<impl Trait>`
                //  - `option::Option<T>::Foo<impl Trait>
                //
                // But not these:
                //  - `<impl Trait>::Foo`
                //  - `option::Option<impl Trait>::Foo`.
                //
                // To implement this, we disallow `impl Trait` from `qself`
                // (for cases like `<impl Trait>::Foo>`)
                // but we allow `impl Trait` in `GenericArgs`
                // iff there are no more PathSegments.
                if let Some(ref qself) = *qself {
                    // `impl Trait` in `qself` is always illegal
                    self.with_banned_impl_trait(|this| this.visit_ty(&qself.ty));
                }

                // Note that there should be a call to visit_path here,
                // so if any logic is added to process `Path`s a call to it should be
                // added both in visit_path and here. This code mirrors visit::walk_path.
                for (i, segment) in path.segments.iter().enumerate() {
                    // Allow `impl Trait` iff we're on the final path segment
                    if i == path.segments.len() - 1 {
                        self.visit_path_segment(path.span, segment);
                    } else {
                        self.with_banned_impl_trait(|this| {
                            this.visit_path_segment(path.span, segment)
                        });
                    }
                }
            }
            _ => visit::walk_ty(self, t),
        }
    }

    fn err_handler(&self) -> &rustc_errors::Handler {
        &self.session.diagnostic()
    }

    fn check_lifetime(&self, ident: Ident) {
        let valid_names = [kw::UnderscoreLifetime, kw::StaticLifetime, kw::Invalid];
        if !valid_names.contains(&ident.name) && ident.without_first_quote().is_reserved() {
            self.err_handler().span_err(ident.span, "lifetimes cannot use keyword names");
        }
    }

    fn check_label(&self, ident: Ident) {
        if ident.without_first_quote().is_reserved() {
            self.err_handler()
                .span_err(ident.span, &format!("invalid label name `{}`", ident.name));
        }
    }

    fn invalid_visibility(&self, vis: &Visibility, note: Option<&str>) {
        if let VisibilityKind::Inherited = vis.node {
            return;
        }

        let mut err =
            struct_span_err!(self.session, vis.span, E0449, "unnecessary visibility qualifier");
        if vis.node.is_pub() {
            err.span_label(vis.span, "`pub` not permitted here because it's implied");
        }
        if let Some(note) = note {
            err.note(note);
        }
        err.emit();
    }

    fn check_decl_no_pat(decl: &FnDecl, mut report_err: impl FnMut(Span, bool)) {
        for Param { pat, .. } in &decl.inputs {
            match pat.kind {
                PatKind::Ident(BindingMode::ByValue(Mutability::Not), _, None) | PatKind::Wild => {}
                PatKind::Ident(BindingMode::ByValue(Mutability::Mut), _, None) => {
                    report_err(pat.span, true)
                }
                _ => report_err(pat.span, false),
            }
        }
    }

    fn check_trait_fn_not_async(&self, span: Span, asyncness: IsAsync) {
        if asyncness.is_async() {
            struct_span_err!(self.session, span, E0706, "trait fns cannot be declared `async`")
                .note("`async` trait functions are not currently supported")
                .note(
                    "consider using the `async-trait` crate: \
                       https://crates.io/crates/async-trait",
                )
                .emit();
        }
    }

    fn check_trait_fn_not_const(&self, constness: Spanned<Constness>) {
        if constness.node == Constness::Const {
            struct_span_err!(
                self.session,
                constness.span,
                E0379,
                "trait fns cannot be declared const"
            )
            .span_label(constness.span, "trait fns cannot be const")
            .emit();
        }
    }

    // FIXME(ecstaticmorse): Instead, use `bound_context` to check this in `visit_param_bound`.
    fn no_questions_in_bounds(&self, bounds: &GenericBounds, where_: &str, is_trait: bool) {
        for bound in bounds {
            if let GenericBound::Trait(ref poly, TraitBoundModifier::Maybe) = *bound {
                let mut err = self.err_handler().struct_span_err(
                    poly.span,
                    &format!("`?Trait` is not permitted in {}", where_),
                );
                if is_trait {
                    let path_str = pprust::path_to_string(&poly.trait_ref.path);
                    err.note(&format!("traits are `?{}` by default", path_str));
                }
                err.emit();
            }
        }
    }

    /// Matches `'-' lit | lit (cf. parser::Parser::parse_literal_maybe_minus)`,
    /// or paths for ranges.
    //
    // FIXME: do we want to allow `expr -> pattern` conversion to create path expressions?
    // That means making this work:
    //
    // ```rust,ignore (FIXME)
    // struct S;
    // macro_rules! m {
    //     ($a:expr) => {
    //         let $a = S;
    //     }
    // }
    // m!(S);
    // ```
    fn check_expr_within_pat(&self, expr: &Expr, allow_paths: bool) {
        match expr.kind {
            ExprKind::Lit(..) | ExprKind::Err => {}
            ExprKind::Path(..) if allow_paths => {}
            ExprKind::Unary(UnOp::Neg, ref inner)
                if match inner.kind {
                    ExprKind::Lit(_) => true,
                    _ => false,
                } => {}
            _ => self.err_handler().span_err(
                expr.span,
                "arbitrary expressions aren't allowed \
                                                         in patterns",
            ),
        }
    }

    fn check_late_bound_lifetime_defs(&self, params: &[GenericParam]) {
        // Check only lifetime parameters are present and that the lifetime
        // parameters that are present have no bounds.
        let non_lt_param_spans: Vec<_> = params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    if !param.bounds.is_empty() {
                        let spans: Vec<_> = param.bounds.iter().map(|b| b.span()).collect();
                        self.err_handler()
                            .span_err(spans, "lifetime bounds cannot be used in this context");
                    }
                    None
                }
                _ => Some(param.ident.span),
            })
            .collect();
        if !non_lt_param_spans.is_empty() {
            self.err_handler().span_err(
                non_lt_param_spans,
                "only lifetime parameters can be used in this context",
            );
        }
    }

    fn check_fn_decl(&self, fn_decl: &FnDecl, self_semantic: SelfSemantic) {
        self.check_decl_cvaradic_pos(fn_decl);
        self.check_decl_attrs(fn_decl);
        self.check_decl_self_param(fn_decl, self_semantic);
    }

    fn check_decl_cvaradic_pos(&self, fn_decl: &FnDecl) {
        match &*fn_decl.inputs {
            [Param { ty, span, .. }] => {
                if let TyKind::CVarArgs = ty.kind {
                    self.err_handler().span_err(
                        *span,
                        "C-variadic function must be declared with at least one named argument",
                    );
                }
            }
            [ps @ .., _] => {
                for Param { ty, span, .. } in ps {
                    if let TyKind::CVarArgs = ty.kind {
                        self.err_handler().span_err(
                            *span,
                            "`...` must be the last argument of a C-variadic function",
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn check_decl_attrs(&self, fn_decl: &FnDecl) {
        fn_decl
            .inputs
            .iter()
            .flat_map(|i| i.attrs.as_ref())
            .filter(|attr| {
                let arr = [sym::allow, sym::cfg, sym::cfg_attr, sym::deny, sym::forbid, sym::warn];
                !arr.contains(&attr.name_or_empty()) && rustc_attr::is_builtin_attr(attr)
            })
            .for_each(|attr| {
                if attr.is_doc_comment() {
                    self.err_handler()
                        .struct_span_err(
                            attr.span,
                            "documentation comments cannot be applied to function parameters",
                        )
                        .span_label(attr.span, "doc comments are not allowed here")
                        .emit();
                } else {
                    self.err_handler().span_err(
                        attr.span,
                        "allow, cfg, cfg_attr, deny, \
                forbid, and warn are the only allowed built-in attributes in function parameters",
                    )
                }
            });
    }

    fn check_decl_self_param(&self, fn_decl: &FnDecl, self_semantic: SelfSemantic) {
        if let (SelfSemantic::No, [param, ..]) = (self_semantic, &*fn_decl.inputs) {
            if param.is_self() {
                self.err_handler()
                    .struct_span_err(
                        param.span,
                        "`self` parameter is only allowed in associated functions",
                    )
                    .span_label(param.span, "not semantically valid as function parameter")
                    .note("associated functions are those in `impl` or `trait` definitions")
                    .emit();
            }
        }
    }

    fn check_defaultness(&self, span: Span, defaultness: Defaultness) {
        if let Defaultness::Default = defaultness {
            self.err_handler()
                .struct_span_err(span, "`default` is only allowed on items in `impl` definitions")
                .emit();
        }
    }

    fn error_item_without_body(&self, sp: Span, ctx: &str, msg: &str, sugg: &str) {
        self.err_handler()
            .struct_span_err(sp, msg)
            .span_suggestion(
                self.session.source_map().end_point(sp),
                &format!("provide a definition for the {}", ctx),
                sugg.to_string(),
                Applicability::HasPlaceholders,
            )
            .emit();
    }

    fn check_impl_item_provided<T>(&self, sp: Span, body: &Option<T>, ctx: &str, sugg: &str) {
        if body.is_none() {
            let msg = format!("associated {} in `impl` without body", ctx);
            self.error_item_without_body(sp, ctx, &msg, sugg);
        }
    }

    fn check_impl_assoc_type_no_bounds(&self, bounds: &[GenericBound]) {
        let span = match bounds {
            [] => return,
            [b0] => b0.span(),
            [b0, .., bl] => b0.span().to(bl.span()),
        };
        self.err_handler()
            .struct_span_err(span, "bounds on associated `type`s in `impl`s have no effect")
            .emit();
    }

    /// An `fn` in `extern { ... }` cannot have a body `{ ... }`.
    fn check_foreign_fn_bodyless(&self, ident: Ident, body: Option<&Block>) {
        let body = match body {
            None => return,
            Some(body) => body,
        };
        self.err_handler()
            .struct_span_err(ident.span, "incorrect function inside `extern` block")
            .span_label(ident.span, "cannot have a body")
            .span_suggestion(
                body.span,
                "remove the invalid body",
                ";".to_string(),
                Applicability::MaybeIncorrect,
            )
            .help(
                "you might have meant to write a function accessible through FFI, \
                which can be done by writing `extern fn` outside of the `extern` block",
            )
            .span_label(
                self.current_extern_span(),
                "`extern` blocks define existing foreign functions and functions \
                inside of them cannot have a body",
            )
            .note("for more information, visit https://doc.rust-lang.org/std/keyword.extern.html")
            .emit();
    }

    fn current_extern_span(&self) -> Span {
        self.session.source_map().def_span(self.extern_mod.unwrap().span)
    }

    /// An `fn` in `extern { ... }` cannot have qualfiers, e.g. `async fn`.
    fn check_foreign_fn_headerless(&self, ident: Ident, span: Span, header: FnHeader) {
        if header.has_qualifiers() {
            self.err_handler()
                .struct_span_err(ident.span, "functions in `extern` blocks cannot have qualifiers")
                .span_label(self.current_extern_span(), "in this `extern` block")
                .span_suggestion(
                    span.until(ident.span.shrink_to_lo()),
                    "remove the qualifiers",
                    "fn ".to_string(),
                    Applicability::MaybeIncorrect,
                )
                .emit();
        }
    }

    /// Reject C-varadic type unless the function is foreign,
    /// or free and `unsafe extern "C"` semantically.
    fn check_c_varadic_type(&self, fk: FnKind<'a>) {
        match (fk.ctxt(), fk.header()) {
            (Some(FnCtxt::Foreign), _) => return,
            (Some(FnCtxt::Free), Some(header)) => match header.ext {
                Extern::Explicit(StrLit { symbol_unescaped: sym::C, .. }) | Extern::Implicit
                    if header.unsafety == Unsafety::Unsafe =>
                {
                    return;
                }
                _ => {}
            },
            _ => {}
        };

        for Param { ty, span, .. } in &fk.decl().inputs {
            if let TyKind::CVarArgs = ty.kind {
                self.err_handler()
                    .struct_span_err(
                        *span,
                        "only foreign or `unsafe extern \"C\" functions may be C-variadic",
                    )
                    .emit();
            }
        }
    }

    /// We currently do not permit const generics in `const fn`,
    /// as this is tantamount to allowing compile-time dependent typing.
    ///
    /// FIXME(const_generics): Is this really true / necessary? Discuss with @varkor.
    /// At any rate, the restriction feels too syntactic. Consider moving it to e.g. typeck.
    fn check_const_fn_const_generic(&self, span: Span, sig: &FnSig, generics: &Generics) {
        if sig.header.constness.node == Constness::Const {
            // Look for const generics and error if we find any.
            for param in &generics.params {
                if let GenericParamKind::Const { .. } = param.kind {
                    self.err_handler()
                        .struct_span_err(span, "const parameters are not permitted in `const fn`")
                        .emit();
                }
            }
        }
    }
}

enum GenericPosition {
    Param,
    Arg,
}

fn validate_generics_order<'a>(
    sess: &Session,
    handler: &rustc_errors::Handler,
    generics: impl Iterator<Item = (ParamKindOrd, Option<&'a [GenericBound]>, Span, Option<String>)>,
    pos: GenericPosition,
    span: Span,
) {
    let mut max_param: Option<ParamKindOrd> = None;
    let mut out_of_order = FxHashMap::default();
    let mut param_idents = vec![];
    let mut found_type = false;
    let mut found_const = false;

    for (kind, bounds, span, ident) in generics {
        if let Some(ident) = ident {
            param_idents.push((kind, bounds, param_idents.len(), ident));
        }
        let max_param = &mut max_param;
        match max_param {
            Some(max_param) if *max_param > kind => {
                let entry = out_of_order.entry(kind).or_insert((*max_param, vec![]));
                entry.1.push(span);
            }
            Some(_) | None => *max_param = Some(kind),
        };
        match kind {
            ParamKindOrd::Type => found_type = true,
            ParamKindOrd::Const => found_const = true,
            _ => {}
        }
    }

    let mut ordered_params = "<".to_string();
    if !out_of_order.is_empty() {
        param_idents.sort_by_key(|&(po, _, i, _)| (po, i));
        let mut first = true;
        for (_, bounds, _, ident) in param_idents {
            if !first {
                ordered_params += ", ";
            }
            ordered_params += &ident;
            if let Some(bounds) = bounds {
                if !bounds.is_empty() {
                    ordered_params += ": ";
                    ordered_params += &pprust::bounds_to_string(&bounds);
                }
            }
            first = false;
        }
    }
    ordered_params += ">";

    let pos_str = match pos {
        GenericPosition::Param => "parameter",
        GenericPosition::Arg => "argument",
    };

    for (param_ord, (max_param, spans)) in &out_of_order {
        let mut err = handler.struct_span_err(
            spans.clone(),
            &format!(
                "{} {pos}s must be declared prior to {} {pos}s",
                param_ord,
                max_param,
                pos = pos_str,
            ),
        );
        if let GenericPosition::Param = pos {
            err.span_suggestion(
                span,
                &format!(
                    "reorder the {}s: lifetimes, then types{}",
                    pos_str,
                    if sess.features_untracked().const_generics { ", then consts" } else { "" },
                ),
                ordered_params.clone(),
                Applicability::MachineApplicable,
            );
        }
        err.emit();
    }

    // FIXME(const_generics): we shouldn't have to abort here at all, but we currently get ICEs
    // if we don't. Const parameters and type parameters can currently conflict if they
    // are out-of-order.
    if !out_of_order.is_empty() && found_type && found_const {
        FatalError.raise();
    }
}

impl<'a> Visitor<'a> for AstValidator<'a> {
    fn visit_attribute(&mut self, attr: &Attribute) {
        validate_attr::check_meta(&self.session.parse_sess, attr);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match &expr.kind {
            ExprKind::InlineAsm(..) if !self.session.target.target.options.allow_asm => {
                struct_span_err!(
                    self.session,
                    expr.span,
                    E0472,
                    "asm! is unsupported on this target"
                )
                .emit();
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.kind {
            TyKind::BareFn(ref bfty) => {
                self.check_fn_decl(&bfty.decl, SelfSemantic::No);
                Self::check_decl_no_pat(&bfty.decl, |span, _| {
                    struct_span_err!(
                        self.session,
                        span,
                        E0561,
                        "patterns aren't allowed in function pointer types"
                    )
                    .emit();
                });
                self.check_late_bound_lifetime_defs(&bfty.generic_params);
            }
            TyKind::TraitObject(ref bounds, ..) => {
                let mut any_lifetime_bounds = false;
                for bound in bounds {
                    if let GenericBound::Outlives(ref lifetime) = *bound {
                        if any_lifetime_bounds {
                            struct_span_err!(
                                self.session,
                                lifetime.ident.span,
                                E0226,
                                "only a single explicit lifetime bound is permitted"
                            )
                            .emit();
                            break;
                        }
                        any_lifetime_bounds = true;
                    }
                }
                self.no_questions_in_bounds(bounds, "trait object types", false);
            }
            TyKind::ImplTrait(_, ref bounds) => {
                if self.is_impl_trait_banned {
                    struct_span_err!(
                        self.session,
                        ty.span,
                        E0667,
                        "`impl Trait` is not allowed in path parameters"
                    )
                    .emit();
                }

                if let Some(outer_impl_trait_sp) = self.outer_impl_trait {
                    struct_span_err!(
                        self.session,
                        ty.span,
                        E0666,
                        "nested `impl Trait` is not allowed"
                    )
                    .span_label(outer_impl_trait_sp, "outer `impl Trait`")
                    .span_label(ty.span, "nested `impl Trait` here")
                    .emit();
                }

                if !bounds
                    .iter()
                    .any(|b| if let GenericBound::Trait(..) = *b { true } else { false })
                {
                    self.err_handler().span_err(ty.span, "at least one trait must be specified");
                }

                self.walk_ty(ty);
                return;
            }
            _ => {}
        }

        self.walk_ty(ty)
    }

    fn visit_label(&mut self, label: &'a Label) {
        self.check_label(label.ident);
        visit::walk_label(self, label);
    }

    fn visit_lifetime(&mut self, lifetime: &'a Lifetime) {
        self.check_lifetime(lifetime.ident);
        visit::walk_lifetime(self, lifetime);
    }

    fn visit_item(&mut self, item: &'a Item) {
        if item.attrs.iter().any(|attr| is_proc_macro_attr(attr)) {
            self.has_proc_macro_decls = true;
        }

        match item.kind {
            ItemKind::Impl {
                unsafety,
                polarity,
                defaultness: _,
                constness: _,
                generics: _,
                of_trait: Some(_),
                ref self_ty,
                items: _,
            } => {
                self.with_in_trait_impl(true, |this| {
                    this.invalid_visibility(&item.vis, None);
                    if let TyKind::Err = self_ty.kind {
                        this.err_handler()
                            .struct_span_err(
                                item.span,
                                "`impl Trait for .. {}` is an obsolete syntax",
                            )
                            .help("use `auto trait Trait {}` instead")
                            .emit();
                    }
                    if unsafety == Unsafety::Unsafe && polarity == ImplPolarity::Negative {
                        struct_span_err!(
                            this.session,
                            item.span,
                            E0198,
                            "negative impls cannot be unsafe"
                        )
                        .emit();
                    }

                    visit::walk_item(this, item);
                });
                return; // Avoid visiting again.
            }
            ItemKind::Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                generics: _,
                of_trait: None,
                self_ty: _,
                items: _,
            } => {
                self.invalid_visibility(
                    &item.vis,
                    Some("place qualifiers on individual impl items instead"),
                );
                if unsafety == Unsafety::Unsafe {
                    struct_span_err!(
                        self.session,
                        item.span,
                        E0197,
                        "inherent impls cannot be unsafe"
                    )
                    .emit();
                }
                if polarity == ImplPolarity::Negative {
                    self.err_handler().span_err(item.span, "inherent impls cannot be negative");
                }
                if defaultness == Defaultness::Default {
                    self.err_handler()
                        .struct_span_err(item.span, "inherent impls cannot be default")
                        .note("only trait implementations may be annotated with default")
                        .emit();
                }
                if constness == Constness::Const {
                    self.err_handler()
                        .struct_span_err(item.span, "inherent impls cannot be `const`")
                        .note("only trait implementations may be annotated with `const`")
                        .emit();
                }
            }
            ItemKind::Fn(ref sig, ref generics, ref body) => {
                self.check_const_fn_const_generic(item.span, sig, generics);

                if body.is_none() {
                    let msg = "free function without a body";
                    self.error_item_without_body(item.span, "function", msg, " { <body> }");
                }
            }
            ItemKind::ForeignMod(_) => {
                let old_item = mem::replace(&mut self.extern_mod, Some(item));
                self.invalid_visibility(
                    &item.vis,
                    Some("place qualifiers on individual foreign items instead"),
                );
                visit::walk_item(self, item);
                self.extern_mod = old_item;
                return; // Avoid visiting again.
            }
            ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    self.invalid_visibility(&variant.vis, None);
                    for field in variant.data.fields() {
                        self.invalid_visibility(&field.vis, None);
                    }
                }
            }
            ItemKind::Trait(is_auto, _, ref generics, ref bounds, ref trait_items) => {
                if is_auto == IsAuto::Yes {
                    // Auto traits cannot have generics, super traits nor contain items.
                    if !generics.params.is_empty() {
                        struct_span_err!(
                            self.session,
                            item.span,
                            E0567,
                            "auto traits cannot have generic parameters"
                        )
                        .emit();
                    }
                    if !bounds.is_empty() {
                        struct_span_err!(
                            self.session,
                            item.span,
                            E0568,
                            "auto traits cannot have super traits"
                        )
                        .emit();
                    }
                    if !trait_items.is_empty() {
                        struct_span_err!(
                            self.session,
                            item.span,
                            E0380,
                            "auto traits cannot have methods or associated items"
                        )
                        .emit();
                    }
                }
                self.no_questions_in_bounds(bounds, "supertraits", true);

                // Equivalent of `visit::walk_item` for `ItemKind::Trait` that inserts a bound
                // context for the supertraits.
                self.visit_vis(&item.vis);
                self.visit_ident(item.ident);
                self.visit_generics(generics);
                self.with_bound_context(BoundContext::TraitBounds, |this| {
                    walk_list!(this, visit_param_bound, bounds);
                });
                walk_list!(self, visit_assoc_item, trait_items, AssocCtxt::Trait);
                walk_list!(self, visit_attribute, &item.attrs);
                return;
            }
            ItemKind::Mod(_) => {
                // Ensure that `path` attributes on modules are recorded as used (cf. issue #35584).
                attr::first_attr_value_str_by_name(&item.attrs, sym::path);
            }
            ItemKind::Union(ref vdata, _) => {
                if let VariantData::Tuple(..) | VariantData::Unit(..) = vdata {
                    self.err_handler()
                        .span_err(item.span, "tuple and unit unions are not permitted");
                }
                if vdata.fields().is_empty() {
                    self.err_handler().span_err(item.span, "unions cannot have zero fields");
                }
            }
            _ => {}
        }

        visit::walk_item(self, item)
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        match &fi.kind {
            ForeignItemKind::Fn(sig, _, body) => {
                self.check_foreign_fn_bodyless(fi.ident, body.as_deref());
                self.check_foreign_fn_headerless(fi.ident, fi.span, sig.header);
            }
            ForeignItemKind::Static(..) | ForeignItemKind::Ty | ForeignItemKind::Macro(..) => {}
        }

        visit::walk_foreign_item(self, fi)
    }

    // Mirrors `visit::walk_generic_args`, but tracks relevant state.
    fn visit_generic_args(&mut self, _: Span, generic_args: &'a GenericArgs) {
        match *generic_args {
            GenericArgs::AngleBracketed(ref data) => {
                walk_list!(self, visit_generic_arg, &data.args);
                validate_generics_order(
                    self.session,
                    self.err_handler(),
                    data.args.iter().map(|arg| {
                        (
                            match arg {
                                GenericArg::Lifetime(..) => ParamKindOrd::Lifetime,
                                GenericArg::Type(..) => ParamKindOrd::Type,
                                GenericArg::Const(..) => ParamKindOrd::Const,
                            },
                            None,
                            arg.span(),
                            None,
                        )
                    }),
                    GenericPosition::Arg,
                    generic_args.span(),
                );

                // Type bindings such as `Item = impl Debug` in `Iterator<Item = Debug>`
                // are allowed to contain nested `impl Trait`.
                self.with_impl_trait(None, |this| {
                    walk_list!(
                        this,
                        visit_assoc_ty_constraint_from_generic_args,
                        &data.constraints
                    );
                });
            }
            GenericArgs::Parenthesized(ref data) => {
                walk_list!(self, visit_ty, &data.inputs);
                if let FunctionRetTy::Ty(ty) = &data.output {
                    // `-> Foo` syntax is essentially an associated type binding,
                    // so it is also allowed to contain nested `impl Trait`.
                    self.with_impl_trait(None, |this| this.visit_ty(ty));
                }
            }
        }
    }

    fn visit_generics(&mut self, generics: &'a Generics) {
        let mut prev_ty_default = None;
        for param in &generics.params {
            if let GenericParamKind::Type { ref default, .. } = param.kind {
                if default.is_some() {
                    prev_ty_default = Some(param.ident.span);
                } else if let Some(span) = prev_ty_default {
                    self.err_handler()
                        .span_err(span, "type parameters with a default must be trailing");
                    break;
                }
            }
        }

        validate_generics_order(
            self.session,
            self.err_handler(),
            generics.params.iter().map(|param| {
                let ident = Some(param.ident.to_string());
                let (kind, ident) = match &param.kind {
                    GenericParamKind::Lifetime { .. } => (ParamKindOrd::Lifetime, ident),
                    GenericParamKind::Type { .. } => (ParamKindOrd::Type, ident),
                    GenericParamKind::Const { ref ty } => {
                        let ty = pprust::ty_to_string(ty);
                        (ParamKindOrd::Const, Some(format!("const {}: {}", param.ident, ty)))
                    }
                };
                (kind, Some(&*param.bounds), param.ident.span, ident)
            }),
            GenericPosition::Param,
            generics.span,
        );

        for predicate in &generics.where_clause.predicates {
            if let WherePredicate::EqPredicate(ref predicate) = *predicate {
                self.err_handler()
                    .struct_span_err(
                        predicate.span,
                        "equality constraints are not yet supported in `where` clauses",
                    )
                    .span_label(predicate.span, "not supported")
                    .note(
                        "see issue #20041 <https://github.com/rust-lang/rust/issues/20041> \
                         for more information",
                    )
                    .emit();
            }
        }

        visit::walk_generics(self, generics)
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        if let GenericParamKind::Lifetime { .. } = param.kind {
            self.check_lifetime(param.ident);
        }
        visit::walk_generic_param(self, param);
    }

    fn visit_param_bound(&mut self, bound: &'a GenericBound) {
        match bound {
            GenericBound::Trait(_, TraitBoundModifier::MaybeConst) => {
                if let Some(ctx) = self.bound_context {
                    let msg = format!("`?const` is not permitted in {}", ctx.description());
                    self.err_handler().span_err(bound.span(), &msg);
                }
            }

            GenericBound::Trait(_, TraitBoundModifier::MaybeConstMaybe) => {
                self.err_handler()
                    .span_err(bound.span(), "`?const` and `?` are mutually exclusive");
            }

            _ => {}
        }

        visit::walk_param_bound(self, bound)
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.kind {
            PatKind::Lit(ref expr) => {
                self.check_expr_within_pat(expr, false);
            }
            PatKind::Range(ref start, ref end, _) => {
                if let Some(expr) = start {
                    self.check_expr_within_pat(expr, true);
                }
                if let Some(expr) = end {
                    self.check_expr_within_pat(expr, true);
                }
            }
            _ => {}
        }

        visit::walk_pat(self, pat)
    }

    fn visit_where_predicate(&mut self, p: &'a WherePredicate) {
        if let &WherePredicate::BoundPredicate(ref bound_predicate) = p {
            // A type binding, eg `for<'c> Foo: Send+Clone+'c`
            self.check_late_bound_lifetime_defs(&bound_predicate.bound_generic_params);
        }
        visit::walk_where_predicate(self, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'a PolyTraitRef, m: &'a TraitBoundModifier) {
        self.check_late_bound_lifetime_defs(&t.bound_generic_params);
        visit::walk_poly_trait_ref(self, t, m);
    }

    fn visit_variant_data(&mut self, s: &'a VariantData) {
        self.with_banned_assoc_ty_bound(|this| visit::walk_struct_def(this, s))
    }

    fn visit_enum_def(
        &mut self,
        enum_definition: &'a EnumDef,
        generics: &'a Generics,
        item_id: NodeId,
        _: Span,
    ) {
        self.with_banned_assoc_ty_bound(|this| {
            visit::walk_enum_def(this, enum_definition, generics, item_id)
        })
    }

    fn visit_fn(&mut self, fk: FnKind<'a>, span: Span, id: NodeId) {
        // Only associated `fn`s can have `self` parameters.
        let self_semantic = match fk.ctxt() {
            Some(FnCtxt::Assoc(_)) => SelfSemantic::Yes,
            _ => SelfSemantic::No,
        };
        self.check_fn_decl(fk.decl(), self_semantic);

        self.check_c_varadic_type(fk);

        // Functions without bodies cannot have patterns.
        if let FnKind::Fn(ctxt, _, sig, _, None) = fk {
            Self::check_decl_no_pat(&sig.decl, |span, mut_ident| {
                let (code, msg, label) = match ctxt {
                    FnCtxt::Foreign => (
                        error_code!(E0130),
                        "patterns aren't allowed in foreign function declarations",
                        "pattern not allowed in foreign function",
                    ),
                    _ => (
                        error_code!(E0642),
                        "patterns aren't allowed in functions without bodies",
                        "pattern not allowed in function without body",
                    ),
                };
                if mut_ident && matches!(ctxt, FnCtxt::Assoc(_)) {
                    self.lint_buffer.buffer_lint(PATTERNS_IN_FNS_WITHOUT_BODY, id, span, msg);
                } else {
                    self.err_handler()
                        .struct_span_err(span, msg)
                        .span_label(span, label)
                        .code(code)
                        .emit();
                }
            });
        }

        visit::walk_fn(self, fk, span);
    }

    fn visit_assoc_item(&mut self, item: &'a AssocItem, ctxt: AssocCtxt) {
        if ctxt == AssocCtxt::Trait {
            self.check_defaultness(item.span, item.defaultness);
        }

        if ctxt == AssocCtxt::Impl {
            match &item.kind {
                AssocItemKind::Const(_, body) => {
                    self.check_impl_item_provided(item.span, body, "constant", " = <expr>;");
                }
                AssocItemKind::Fn(_, body) => {
                    self.check_impl_item_provided(item.span, body, "function", " { <body> }");
                }
                AssocItemKind::TyAlias(bounds, body) => {
                    self.check_impl_item_provided(item.span, body, "type", " = <type>;");
                    self.check_impl_assoc_type_no_bounds(bounds);
                }
                _ => {}
            }
        }

        if ctxt == AssocCtxt::Trait || self.in_trait_impl {
            self.invalid_visibility(&item.vis, None);
            if let AssocItemKind::Fn(sig, _) = &item.kind {
                self.check_trait_fn_not_const(sig.header.constness);
                self.check_trait_fn_not_async(item.span, sig.header.asyncness.node);
            }
        }

        self.with_in_trait_impl(false, |this| visit::walk_assoc_item(this, item, ctxt));
    }
}

pub fn check_crate(session: &Session, krate: &Crate, lints: &mut LintBuffer) -> bool {
    let mut validator = AstValidator {
        session,
        extern_mod: None,
        in_trait_impl: false,
        has_proc_macro_decls: false,
        outer_impl_trait: None,
        bound_context: None,
        is_impl_trait_banned: false,
        is_assoc_ty_bound_banned: false,
        lint_buffer: lints,
    };
    visit::walk_crate(&mut validator, krate);

    validator.has_proc_macro_decls
}
