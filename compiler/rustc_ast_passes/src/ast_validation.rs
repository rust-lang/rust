// Validate AST before lowering it to HIR.
//
// This pass is supposed to catch things that fit into AST data structures,
// but not permitted by the language. It runs after expansion when AST is frozen,
// so it can check for erroneous constructions produced by syntax extensions.
// This pass is supposed to perform only simple checks not requiring name resolution
// or type checking or some other kind of complex analysis.

use itertools::{Either, Itertools};
use rustc_ast::ptr::P;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::walk_list;
use rustc_ast::*;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{error_code, pluralize, struct_span_err, Applicability};
use rustc_parse::validate_attr;
use rustc_session::lint::builtin::{MISSING_ABI, PATTERNS_IN_FNS_WITHOUT_BODY};
use rustc_session::lint::{BuiltinLintDiagnostics, LintBuffer};
use rustc_session::Session;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use rustc_target::spec::abi;
use std::mem;
use std::ops::DerefMut;

const MORE_EXTERN: &str =
    "for more information, visit https://doc.rust-lang.org/std/keyword.extern.html";

/// Is `self` allowed semantically as the first parameter in an `FnDecl`?
enum SelfSemantic {
    Yes,
    No,
}

struct AstValidator<'a> {
    session: &'a Session,

    /// The span of the `extern` in an `extern { ... }` block, if any.
    extern_mod: Option<&'a Item>,

    /// Are we inside a trait impl?
    in_trait_impl: bool,

    in_const_trait_impl: bool,

    has_proc_macro_decls: bool,

    /// Used to ban nested `impl Trait`, e.g., `impl Into<impl Debug>`.
    /// Nested `impl Trait` _is_ allowed in associated type position,
    /// e.g., `impl Iterator<Item = impl Debug>`.
    outer_impl_trait: Option<Span>,

    is_tilde_const_allowed: bool,

    /// Used to ban `impl Trait` in path projections like `<impl Iterator>::Item`
    /// or `Foo::Bar<impl Trait>`
    is_impl_trait_banned: bool,

    /// Used to ban associated type bounds (i.e., `Type<AssocType: Bounds>`) in
    /// certain positions.
    is_assoc_ty_bound_banned: bool,

    /// Used to allow `let` expressions in certain syntactic locations.
    is_let_allowed: bool,

    lint_buffer: &'a mut LintBuffer,
}

impl<'a> AstValidator<'a> {
    fn with_in_trait_impl(
        &mut self,
        is_in: bool,
        constness: Option<Const>,
        f: impl FnOnce(&mut Self),
    ) {
        let old = mem::replace(&mut self.in_trait_impl, is_in);
        let old_const =
            mem::replace(&mut self.in_const_trait_impl, matches!(constness, Some(Const::Yes(_))));
        f(self);
        self.in_trait_impl = old;
        self.in_const_trait_impl = old_const;
    }

    fn with_banned_impl_trait(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_impl_trait_banned, true);
        f(self);
        self.is_impl_trait_banned = old;
    }

    fn with_tilde_const_allowed(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_tilde_const_allowed, true);
        f(self);
        self.is_tilde_const_allowed = old;
    }

    fn with_banned_tilde_const(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_tilde_const_allowed, false);
        f(self);
        self.is_tilde_const_allowed = old;
    }

    fn with_let_allowed(&mut self, allowed: bool, f: impl FnOnce(&mut Self, bool)) {
        let old = mem::replace(&mut self.is_let_allowed, allowed);
        f(self, old);
        self.is_let_allowed = old;
    }

    /// Emits an error banning the `let` expression provided in the given location.
    fn ban_let_expr(&self, expr: &'a Expr) {
        let sess = &self.session;
        if sess.opts.unstable_features.is_nightly_build() {
            sess.struct_span_err(expr.span, "`let` expressions are not supported here")
                .note("only supported directly in conditions of `if`- and `while`-expressions")
                .note("as well as when nested within `&&` and parentheses in those conditions")
                .emit();
        } else {
            sess.struct_span_err(expr.span, "expected expression, found statement (`let`)")
                .note("variable declaration using `let` is a statement")
                .emit();
        }
    }

    fn with_banned_assoc_ty_bound(&mut self, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.is_assoc_ty_bound_banned, true);
        f(self);
        self.is_assoc_ty_bound_banned = old;
    }

    fn with_impl_trait(&mut self, outer: Option<Span>, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.outer_impl_trait, outer);
        if outer.is_some() {
            self.with_banned_tilde_const(f);
        } else {
            f(self);
        }
        self.outer_impl_trait = old;
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
            TyKind::TraitObject(..) => self.with_banned_tilde_const(|this| visit::walk_ty(this, t)),
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

    fn visit_struct_field_def(&mut self, field: &'a FieldDef) {
        if let Some(ident) = field.ident {
            if ident.name == kw::Underscore {
                self.visit_vis(&field.vis);
                self.visit_ident(ident);
                self.visit_ty_common(&field.ty);
                self.walk_ty(&field.ty);
                walk_list!(self, visit_attribute, &field.attrs);
                return;
            }
        }
        self.visit_field_def(field);
    }

    fn err_handler(&self) -> &rustc_errors::Handler {
        &self.session.diagnostic()
    }

    fn check_lifetime(&self, ident: Ident) {
        let valid_names = [kw::UnderscoreLifetime, kw::StaticLifetime, kw::Empty];
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
        if let VisibilityKind::Inherited = vis.kind {
            return;
        }

        let mut err =
            struct_span_err!(self.session, vis.span, E0449, "unnecessary visibility qualifier");
        if vis.kind.is_pub() {
            err.span_label(vis.span, "`pub` not permitted here because it's implied");
        }
        if let Some(note) = note {
            err.note(note);
        }
        err.emit();
    }

    fn check_decl_no_pat(decl: &FnDecl, mut report_err: impl FnMut(Span, Option<Ident>, bool)) {
        for Param { pat, .. } in &decl.inputs {
            match pat.kind {
                PatKind::Ident(BindingMode::ByValue(Mutability::Not), _, None) | PatKind::Wild => {}
                PatKind::Ident(BindingMode::ByValue(Mutability::Mut), ident, None) => {
                    report_err(pat.span, Some(ident), true)
                }
                _ => report_err(pat.span, None, false),
            }
        }
    }

    fn check_trait_fn_not_async(&self, fn_span: Span, asyncness: Async) {
        if let Async::Yes { span, .. } = asyncness {
            struct_span_err!(
                self.session,
                fn_span,
                E0706,
                "functions in traits cannot be declared `async`"
            )
            .span_label(span, "`async` because of this")
            .note("`async` trait functions are not currently supported")
            .note("consider using the `async-trait` crate: https://crates.io/crates/async-trait")
            .emit();
        }
    }

    fn check_trait_fn_not_const(&self, constness: Const) {
        if let Const::Yes(span) = constness {
            struct_span_err!(
                self.session,
                span,
                E0379,
                "functions in traits cannot be declared const"
            )
            .span_label(span, "functions in traits cannot be const")
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
            ExprKind::Lit(..) | ExprKind::ConstBlock(..) | ExprKind::Err => {}
            ExprKind::Path(..) if allow_paths => {}
            ExprKind::Unary(UnOp::Neg, ref inner) if matches!(inner.kind, ExprKind::Lit(_)) => {}
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
        self.check_decl_num_args(fn_decl);
        self.check_decl_cvaradic_pos(fn_decl);
        self.check_decl_attrs(fn_decl);
        self.check_decl_self_param(fn_decl, self_semantic);
    }

    /// Emits fatal error if function declaration has more than `u16::MAX` arguments
    /// Error is fatal to prevent errors during typechecking
    fn check_decl_num_args(&self, fn_decl: &FnDecl) {
        let max_num_args: usize = u16::MAX.into();
        if fn_decl.inputs.len() > max_num_args {
            let Param { span, .. } = fn_decl.inputs[0];
            self.err_handler().span_fatal(
                span,
                &format!("function can not have more than {} arguments", max_num_args),
            );
        }
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
        if let Defaultness::Default(def_span) = defaultness {
            let span = self.session.source_map().guess_head_span(span);
            self.err_handler()
                .struct_span_err(span, "`default` is only allowed on items in trait impls")
                .span_label(def_span, "`default` because of this")
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

    fn check_type_no_bounds(&self, bounds: &[GenericBound], ctx: &str) {
        let span = match bounds {
            [] => return,
            [b0] => b0.span(),
            [b0, .., bl] => b0.span().to(bl.span()),
        };
        self.err_handler()
            .struct_span_err(span, &format!("bounds on `type`s in {} have no effect", ctx))
            .emit();
    }

    fn check_foreign_ty_genericless(&self, generics: &Generics) {
        let cannot_have = |span, descr, remove_descr| {
            self.err_handler()
                .struct_span_err(
                    span,
                    &format!("`type`s inside `extern` blocks cannot have {}", descr),
                )
                .span_suggestion(
                    span,
                    &format!("remove the {}", remove_descr),
                    String::new(),
                    Applicability::MaybeIncorrect,
                )
                .span_label(self.current_extern_span(), "`extern` block begins here")
                .note(MORE_EXTERN)
                .emit();
        };

        if !generics.params.is_empty() {
            cannot_have(generics.span, "generic parameters", "generic parameters");
        }

        if !generics.where_clause.predicates.is_empty() {
            cannot_have(generics.where_clause.span, "`where` clauses", "`where` clause");
        }
    }

    fn check_foreign_kind_bodyless(&self, ident: Ident, kind: &str, body: Option<Span>) {
        let body = match body {
            None => return,
            Some(body) => body,
        };
        self.err_handler()
            .struct_span_err(ident.span, &format!("incorrect `{}` inside `extern` block", kind))
            .span_label(ident.span, "cannot have a body")
            .span_label(body, "the invalid body")
            .span_label(
                self.current_extern_span(),
                format!(
                    "`extern` blocks define existing foreign {0}s and {0}s \
                    inside of them cannot have a body",
                    kind
                ),
            )
            .note(MORE_EXTERN)
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
            .note(MORE_EXTERN)
            .emit();
    }

    fn current_extern_span(&self) -> Span {
        self.session.source_map().guess_head_span(self.extern_mod.unwrap().span)
    }

    /// An `fn` in `extern { ... }` cannot have qualifiers, e.g. `async fn`.
    fn check_foreign_fn_headerless(&self, ident: Ident, span: Span, header: FnHeader) {
        if header.has_qualifiers() {
            self.err_handler()
                .struct_span_err(ident.span, "functions in `extern` blocks cannot have qualifiers")
                .span_label(self.current_extern_span(), "in this `extern` block")
                .span_suggestion_verbose(
                    span.until(ident.span.shrink_to_lo()),
                    "remove the qualifiers",
                    "fn ".to_string(),
                    Applicability::MaybeIncorrect,
                )
                .emit();
        }
    }

    /// An item in `extern { ... }` cannot use non-ascii identifier.
    fn check_foreign_item_ascii_only(&self, ident: Ident) {
        let symbol_str = ident.as_str();
        if !symbol_str.is_ascii() {
            let n = 83942;
            self.err_handler()
                .struct_span_err(
                    ident.span,
                    "items in `extern` blocks cannot use non-ascii identifiers",
                )
                .span_label(self.current_extern_span(), "in this `extern` block")
                .note(&format!(
                    "this limitation may be lifted in the future; see issue #{} <https://github.com/rust-lang/rust/issues/{}> for more information",
                    n, n,
                ))
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
                    if matches!(header.unsafety, Unsafe::Yes(_)) =>
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
                        "only foreign or `unsafe extern \"C\"` functions may be C-variadic",
                    )
                    .emit();
            }
        }
    }

    fn check_item_named(&self, ident: Ident, kind: &str) {
        if ident.name != kw::Underscore {
            return;
        }
        self.err_handler()
            .struct_span_err(ident.span, &format!("`{}` items in this context need a name", kind))
            .span_label(ident.span, format!("`_` is not a valid name for this `{}` item", kind))
            .emit();
    }

    fn check_nomangle_item_asciionly(&self, ident: Ident, item_span: Span) {
        if ident.name.as_str().is_ascii() {
            return;
        }
        let head_span = self.session.source_map().guess_head_span(item_span);
        struct_span_err!(
            self.session,
            head_span,
            E0754,
            "`#[no_mangle]` requires ASCII identifier"
        )
        .emit();
    }

    fn check_mod_file_item_asciionly(&self, ident: Ident) {
        if ident.name.as_str().is_ascii() {
            return;
        }
        struct_span_err!(
            self.session,
            ident.span,
            E0754,
            "trying to load file for module `{}` with non-ascii identifier name",
            ident.name
        )
        .help("consider using `#[path]` attribute to specify filesystem path")
        .emit();
    }

    fn deny_generic_params(&self, generics: &Generics, ident_span: Span) {
        if !generics.params.is_empty() {
            struct_span_err!(
                self.session,
                generics.span,
                E0567,
                "auto traits cannot have generic parameters"
            )
            .span_label(ident_span, "auto trait cannot have generic parameters")
            .span_suggestion(
                generics.span,
                "remove the parameters",
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
        }
    }

    fn emit_e0568(&self, span: Span, ident_span: Span) {
        struct_span_err!(
            self.session,
            span,
            E0568,
            "auto traits cannot have super traits or lifetime bounds"
        )
        .span_label(ident_span, "auto trait cannot have super traits or lifetime bounds")
        .span_suggestion(
            span,
            "remove the super traits or lifetime bounds",
            String::new(),
            Applicability::MachineApplicable,
        )
        .emit();
    }

    fn deny_super_traits(&self, bounds: &GenericBounds, ident_span: Span) {
        if let [.., last] = &bounds[..] {
            let span = ident_span.shrink_to_hi().to(last.span());
            self.emit_e0568(span, ident_span);
        }
    }

    fn deny_where_clause(&self, where_clause: &WhereClause, ident_span: Span) {
        if !where_clause.predicates.is_empty() {
            self.emit_e0568(where_clause.span, ident_span);
        }
    }

    fn deny_items(&self, trait_items: &[P<AssocItem>], ident_span: Span) {
        if !trait_items.is_empty() {
            let spans: Vec<_> = trait_items.iter().map(|i| i.ident.span).collect();
            let total_span = trait_items.first().unwrap().span.to(trait_items.last().unwrap().span);
            struct_span_err!(
                self.session,
                spans,
                E0380,
                "auto traits cannot have associated items"
            )
            .span_suggestion(
                total_span,
                "remove these associated items",
                String::new(),
                Applicability::MachineApplicable,
            )
            .span_label(ident_span, "auto trait cannot have associated items")
            .emit();
        }
    }

    fn correct_generic_order_suggestion(&self, data: &AngleBracketedArgs) -> String {
        // Lifetimes always come first.
        let lt_sugg = data.args.iter().filter_map(|arg| match arg {
            AngleBracketedArg::Arg(lt @ GenericArg::Lifetime(_)) => {
                Some(pprust::to_string(|s| s.print_generic_arg(lt)))
            }
            _ => None,
        });
        let args_sugg = data.args.iter().filter_map(|a| match a {
            AngleBracketedArg::Arg(GenericArg::Lifetime(_)) | AngleBracketedArg::Constraint(_) => {
                None
            }
            AngleBracketedArg::Arg(arg) => Some(pprust::to_string(|s| s.print_generic_arg(arg))),
        });
        // Constraints always come last.
        let constraint_sugg = data.args.iter().filter_map(|a| match a {
            AngleBracketedArg::Arg(_) => None,
            AngleBracketedArg::Constraint(c) => {
                Some(pprust::to_string(|s| s.print_assoc_constraint(c)))
            }
        });
        format!(
            "<{}>",
            lt_sugg.chain(args_sugg).chain(constraint_sugg).collect::<Vec<String>>().join(", ")
        )
    }

    /// Enforce generic args coming before constraints in `<...>` of a path segment.
    fn check_generic_args_before_constraints(&self, data: &AngleBracketedArgs) {
        // Early exit in case it's partitioned as it should be.
        if data.args.iter().is_partitioned(|arg| matches!(arg, AngleBracketedArg::Arg(_))) {
            return;
        }
        // Find all generic argument coming after the first constraint...
        let (constraint_spans, arg_spans): (Vec<Span>, Vec<Span>) =
            data.args.iter().partition_map(|arg| match arg {
                AngleBracketedArg::Constraint(c) => Either::Left(c.span),
                AngleBracketedArg::Arg(a) => Either::Right(a.span()),
            });
        let args_len = arg_spans.len();
        let constraint_len = constraint_spans.len();
        // ...and then error:
        self.err_handler()
            .struct_span_err(
                arg_spans.clone(),
                "generic arguments must come before the first constraint",
            )
            .span_label(constraint_spans[0], &format!("constraint{}", pluralize!(constraint_len)))
            .span_label(
                *arg_spans.iter().last().unwrap(),
                &format!("generic argument{}", pluralize!(args_len)),
            )
            .span_labels(constraint_spans, "")
            .span_labels(arg_spans, "")
            .span_suggestion_verbose(
                data.span,
                &format!(
                    "move the constraint{} after the generic argument{}",
                    pluralize!(constraint_len),
                    pluralize!(args_len)
                ),
                self.correct_generic_order_suggestion(&data),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    fn visit_ty_common(&mut self, ty: &'a Ty) {
        match ty.kind {
            TyKind::BareFn(ref bfty) => {
                self.check_fn_decl(&bfty.decl, SelfSemantic::No);
                Self::check_decl_no_pat(&bfty.decl, |span, _, _| {
                    struct_span_err!(
                        self.session,
                        span,
                        E0561,
                        "patterns aren't allowed in function pointer types"
                    )
                    .emit();
                });
                self.check_late_bound_lifetime_defs(&bfty.generic_params);
                if let Extern::Implicit = bfty.ext {
                    let sig_span = self.session.source_map().next_point(ty.span.shrink_to_lo());
                    self.maybe_lint_missing_abi(sig_span, ty.id);
                }
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

                if !bounds.iter().any(|b| matches!(b, GenericBound::Trait(..))) {
                    self.err_handler().span_err(ty.span, "at least one trait must be specified");
                }
            }
            _ => {}
        }
    }

    fn maybe_lint_missing_abi(&mut self, span: Span, id: NodeId) {
        // FIXME(davidtwco): This is a hack to detect macros which produce spans of the
        // call site which do not have a macro backtrace. See #61963.
        let is_macro_callsite = self
            .session
            .source_map()
            .span_to_snippet(span)
            .map(|snippet| snippet.starts_with("#["))
            .unwrap_or(true);
        if !is_macro_callsite {
            self.lint_buffer.buffer_lint_with_diagnostic(
                MISSING_ABI,
                id,
                span,
                "extern declarations without an explicit ABI are deprecated",
                BuiltinLintDiagnostics::MissingAbi(span, abi::Abi::FALLBACK),
            )
        }
    }
}

/// Checks that generic parameters are in the correct order,
/// which is lifetimes, then types and then consts. (`<'a, T, const N: usize>`)
fn validate_generic_param_order(
    sess: &Session,
    handler: &rustc_errors::Handler,
    generics: &[GenericParam],
    span: Span,
) {
    let mut max_param: Option<ParamKindOrd> = None;
    let mut out_of_order = FxHashMap::default();
    let mut param_idents = Vec::with_capacity(generics.len());

    for (idx, param) in generics.iter().enumerate() {
        let ident = param.ident;
        let (kind, bounds, span) = (&param.kind, &param.bounds, ident.span);
        let (ord_kind, ident) = match &param.kind {
            GenericParamKind::Lifetime => (ParamKindOrd::Lifetime, ident.to_string()),
            GenericParamKind::Type { default: _ } => (ParamKindOrd::Type, ident.to_string()),
            GenericParamKind::Const { ref ty, kw_span: _, default: _ } => {
                let ty = pprust::ty_to_string(ty);
                let unordered = sess.features_untracked().unordered_const_ty_params();
                (ParamKindOrd::Const { unordered }, format!("const {}: {}", ident, ty))
            }
        };
        param_idents.push((kind, ord_kind, bounds, idx, ident));
        match max_param {
            Some(max_param) if max_param > ord_kind => {
                let entry = out_of_order.entry(ord_kind).or_insert((max_param, vec![]));
                entry.1.push(span);
            }
            Some(_) | None => max_param = Some(ord_kind),
        };
    }

    if !out_of_order.is_empty() {
        let mut ordered_params = "<".to_string();
        param_idents.sort_by_key(|&(_, po, _, i, _)| (po, i));
        let mut first = true;
        for (kind, _, bounds, _, ident) in param_idents {
            if !first {
                ordered_params += ", ";
            }
            ordered_params += &ident;

            if !bounds.is_empty() {
                ordered_params += ": ";
                ordered_params += &pprust::bounds_to_string(&bounds);
            }

            match kind {
                GenericParamKind::Type { default: Some(default) } => {
                    ordered_params += " = ";
                    ordered_params += &pprust::ty_to_string(default);
                }
                GenericParamKind::Type { default: None } => (),
                GenericParamKind::Lifetime => (),
                GenericParamKind::Const { ty: _, kw_span: _, default: Some(default) } => {
                    ordered_params += " = ";
                    ordered_params += &pprust::expr_to_string(&*default.value);
                }
                GenericParamKind::Const { ty: _, kw_span: _, default: None } => (),
            }
            first = false;
        }

        ordered_params += ">";

        for (param_ord, (max_param, spans)) in &out_of_order {
            let mut err = handler.struct_span_err(
                spans.clone(),
                &format!(
                    "{} parameters must be declared prior to {} parameters",
                    param_ord, max_param,
                ),
            );
            err.span_suggestion(
                span,
                &format!(
                    "reorder the parameters: lifetimes, {}",
                    if sess.features_untracked().unordered_const_ty_params() {
                        "then consts and types"
                    } else {
                        "then types, then consts"
                    }
                ),
                ordered_params.clone(),
                Applicability::MachineApplicable,
            );
            err.emit();
        }
    }
}

impl<'a> Visitor<'a> for AstValidator<'a> {
    fn visit_attribute(&mut self, attr: &Attribute) {
        validate_attr::check_meta(&self.session.parse_sess, attr);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        self.with_let_allowed(false, |this, let_allowed| match &expr.kind {
            ExprKind::If(cond, then, opt_else) => {
                this.visit_block(then);
                walk_list!(this, visit_expr, opt_else);
                this.with_let_allowed(true, |this, _| this.visit_expr(cond));
                return;
            }
            ExprKind::Let(..) if !let_allowed => this.ban_let_expr(expr),
            ExprKind::LlvmInlineAsm(..) if !this.session.target.allow_asm => {
                struct_span_err!(
                    this.session,
                    expr.span,
                    E0472,
                    "llvm_asm! is unsupported on this target"
                )
                .emit();
            }
            ExprKind::Match(expr, arms) => {
                this.visit_expr(expr);
                for arm in arms {
                    this.visit_expr(&arm.body);
                    this.visit_pat(&arm.pat);
                    walk_list!(this, visit_attribute, &arm.attrs);
                    if let Some(ref guard) = arm.guard {
                        if let ExprKind::Let(_, ref expr, _) = guard.kind {
                            this.with_let_allowed(true, |this, _| this.visit_expr(expr));
                            return;
                        }
                    }
                }
            }
            ExprKind::Paren(_) | ExprKind::Binary(Spanned { node: BinOpKind::And, .. }, ..) => {
                this.with_let_allowed(let_allowed, |this, _| visit::walk_expr(this, expr));
                return;
            }
            ExprKind::While(cond, then, opt_label) => {
                walk_list!(this, visit_label, opt_label);
                this.visit_block(then);
                this.with_let_allowed(true, |this, _| this.visit_expr(cond));
                return;
            }
            _ => visit::walk_expr(this, expr),
        });
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        self.visit_ty_common(ty);
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

    fn visit_field_def(&mut self, s: &'a FieldDef) {
        visit::walk_field_def(self, s)
    }

    fn visit_item(&mut self, item: &'a Item) {
        if item.attrs.iter().any(|attr| self.session.is_proc_macro_attr(attr)) {
            self.has_proc_macro_decls = true;
        }

        if self.session.contains_name(&item.attrs, sym::no_mangle) {
            self.check_nomangle_item_asciionly(item.ident, item.span);
        }

        match item.kind {
            ItemKind::Impl(box Impl {
                unsafety,
                polarity,
                defaultness: _,
                constness,
                ref generics,
                of_trait: Some(ref t),
                ref self_ty,
                ref items,
            }) => {
                self.with_in_trait_impl(true, Some(constness), |this| {
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
                    if let (Unsafe::Yes(span), ImplPolarity::Negative(sp)) = (unsafety, polarity) {
                        struct_span_err!(
                            this.session,
                            sp.to(t.path.span),
                            E0198,
                            "negative impls cannot be unsafe"
                        )
                        .span_label(sp, "negative because of this")
                        .span_label(span, "unsafe because of this")
                        .emit();
                    }

                    this.visit_vis(&item.vis);
                    this.visit_ident(item.ident);
                    if let Const::Yes(_) = constness {
                        this.with_tilde_const_allowed(|this| this.visit_generics(generics));
                    } else {
                        this.visit_generics(generics);
                    }
                    this.visit_trait_ref(t);
                    this.visit_ty(self_ty);

                    walk_list!(this, visit_assoc_item, items, AssocCtxt::Impl);
                });
                return; // Avoid visiting again.
            }
            ItemKind::Impl(box Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                generics: _,
                of_trait: None,
                ref self_ty,
                items: _,
            }) => {
                let error = |annotation_span, annotation| {
                    let mut err = self.err_handler().struct_span_err(
                        self_ty.span,
                        &format!("inherent impls cannot be {}", annotation),
                    );
                    err.span_label(annotation_span, &format!("{} because of this", annotation));
                    err.span_label(self_ty.span, "inherent impl for this type");
                    err
                };

                self.invalid_visibility(
                    &item.vis,
                    Some("place qualifiers on individual impl items instead"),
                );
                if let Unsafe::Yes(span) = unsafety {
                    error(span, "unsafe").code(error_code!(E0197)).emit();
                }
                if let ImplPolarity::Negative(span) = polarity {
                    error(span, "negative").emit();
                }
                if let Defaultness::Default(def_span) = defaultness {
                    error(def_span, "`default`")
                        .note("only trait implementations may be annotated with `default`")
                        .emit();
                }
                if let Const::Yes(span) = constness {
                    error(span, "`const`")
                        .note("only trait implementations may be annotated with `const`")
                        .emit();
                }
            }
            ItemKind::Fn(box Fn { defaultness, ref sig, ref generics, ref body }) => {
                self.check_defaultness(item.span, defaultness);

                if body.is_none() {
                    let msg = "free function without a body";
                    self.error_item_without_body(item.span, "function", msg, " { <body> }");
                }
                self.visit_vis(&item.vis);
                self.visit_ident(item.ident);
                if let Const::Yes(_) = sig.header.constness {
                    self.with_tilde_const_allowed(|this| this.visit_generics(generics));
                } else {
                    self.visit_generics(generics);
                }
                let kind = FnKind::Fn(FnCtxt::Free, item.ident, sig, &item.vis, body.as_deref());
                self.visit_fn(kind, item.span, item.id);
                walk_list!(self, visit_attribute, &item.attrs);
                return; // Avoid visiting again.
            }
            ItemKind::ForeignMod(ForeignMod { abi, unsafety, .. }) => {
                let old_item = mem::replace(&mut self.extern_mod, Some(item));
                self.invalid_visibility(
                    &item.vis,
                    Some("place qualifiers on individual foreign items instead"),
                );
                if let Unsafe::Yes(span) = unsafety {
                    self.err_handler().span_err(span, "extern block cannot be declared unsafe");
                }
                if abi.is_none() {
                    self.maybe_lint_missing_abi(item.span, item.id);
                }
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
            ItemKind::Trait(box Trait { is_auto, ref generics, ref bounds, ref items, .. }) => {
                if is_auto == IsAuto::Yes {
                    // Auto traits cannot have generics, super traits nor contain items.
                    self.deny_generic_params(generics, item.ident.span);
                    self.deny_super_traits(bounds, item.ident.span);
                    self.deny_where_clause(&generics.where_clause, item.ident.span);
                    self.deny_items(items, item.ident.span);
                }
                self.no_questions_in_bounds(bounds, "supertraits", true);

                // Equivalent of `visit::walk_item` for `ItemKind::Trait` that inserts a bound
                // context for the supertraits.
                self.visit_vis(&item.vis);
                self.visit_ident(item.ident);
                self.visit_generics(generics);
                self.with_banned_tilde_const(|this| walk_list!(this, visit_param_bound, bounds));
                walk_list!(self, visit_assoc_item, items, AssocCtxt::Trait);
                walk_list!(self, visit_attribute, &item.attrs);
                return;
            }
            ItemKind::Mod(unsafety, ref mod_kind) => {
                if let Unsafe::Yes(span) = unsafety {
                    self.err_handler().span_err(span, "module cannot be declared unsafe");
                }
                // Ensure that `path` attributes on modules are recorded as used (cf. issue #35584).
                if !matches!(mod_kind, ModKind::Loaded(_, Inline::Yes, _))
                    && !self.session.contains_name(&item.attrs, sym::path)
                {
                    self.check_mod_file_item_asciionly(item.ident);
                }
            }
            ItemKind::Struct(ref vdata, ref generics) => match vdata {
                // Duplicating the `Visitor` logic allows catching all cases
                // of `Anonymous(Struct, Union)` outside of a field struct or union.
                //
                // Inside `visit_ty` the validator catches every `Anonymous(Struct, Union)` it
                // encounters, and only on `ItemKind::Struct` and `ItemKind::Union`
                // it uses `visit_ty_common`, which doesn't contain that specific check.
                VariantData::Struct(ref fields, ..) => {
                    self.visit_vis(&item.vis);
                    self.visit_ident(item.ident);
                    self.visit_generics(generics);
                    self.with_banned_assoc_ty_bound(|this| {
                        walk_list!(this, visit_struct_field_def, fields);
                    });
                    walk_list!(self, visit_attribute, &item.attrs);
                    return;
                }
                _ => {}
            },
            ItemKind::Union(ref vdata, ref generics) => {
                if vdata.fields().is_empty() {
                    self.err_handler().span_err(item.span, "unions cannot have zero fields");
                }
                match vdata {
                    VariantData::Struct(ref fields, ..) => {
                        self.visit_vis(&item.vis);
                        self.visit_ident(item.ident);
                        self.visit_generics(generics);
                        self.with_banned_assoc_ty_bound(|this| {
                            walk_list!(this, visit_struct_field_def, fields);
                        });
                        walk_list!(self, visit_attribute, &item.attrs);
                        return;
                    }
                    _ => {}
                }
            }
            ItemKind::Const(def, .., None) => {
                self.check_defaultness(item.span, def);
                let msg = "free constant item without body";
                self.error_item_without_body(item.span, "constant", msg, " = <expr>;");
            }
            ItemKind::Static(.., None) => {
                let msg = "free static item without body";
                self.error_item_without_body(item.span, "static", msg, " = <expr>;");
            }
            ItemKind::TyAlias(box TyAlias { defaultness, ref bounds, ref ty, .. }) => {
                self.check_defaultness(item.span, defaultness);
                if ty.is_none() {
                    let msg = "free type alias without body";
                    self.error_item_without_body(item.span, "type", msg, " = <type>;");
                }
                self.check_type_no_bounds(bounds, "this context");
            }
            _ => {}
        }

        visit::walk_item(self, item);
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        match &fi.kind {
            ForeignItemKind::Fn(box Fn { defaultness, sig, body, .. }) => {
                self.check_defaultness(fi.span, *defaultness);
                self.check_foreign_fn_bodyless(fi.ident, body.as_deref());
                self.check_foreign_fn_headerless(fi.ident, fi.span, sig.header);
                self.check_foreign_item_ascii_only(fi.ident);
            }
            ForeignItemKind::TyAlias(box TyAlias { defaultness, generics, bounds, ty, .. }) => {
                self.check_defaultness(fi.span, *defaultness);
                self.check_foreign_kind_bodyless(fi.ident, "type", ty.as_ref().map(|b| b.span));
                self.check_type_no_bounds(bounds, "`extern` blocks");
                self.check_foreign_ty_genericless(generics);
                self.check_foreign_item_ascii_only(fi.ident);
            }
            ForeignItemKind::Static(_, _, body) => {
                self.check_foreign_kind_bodyless(fi.ident, "static", body.as_ref().map(|b| b.span));
                self.check_foreign_item_ascii_only(fi.ident);
            }
            ForeignItemKind::MacCall(..) => {}
        }

        visit::walk_foreign_item(self, fi)
    }

    // Mirrors `visit::walk_generic_args`, but tracks relevant state.
    fn visit_generic_args(&mut self, _: Span, generic_args: &'a GenericArgs) {
        match *generic_args {
            GenericArgs::AngleBracketed(ref data) => {
                self.check_generic_args_before_constraints(data);

                for arg in &data.args {
                    match arg {
                        AngleBracketedArg::Arg(arg) => self.visit_generic_arg(arg),
                        // Type bindings such as `Item = impl Debug` in `Iterator<Item = Debug>`
                        // are allowed to contain nested `impl Trait`.
                        AngleBracketedArg::Constraint(constraint) => {
                            self.with_impl_trait(None, |this| {
                                this.visit_assoc_ty_constraint_from_generic_args(constraint);
                            });
                        }
                    }
                }
            }
            GenericArgs::Parenthesized(ref data) => {
                walk_list!(self, visit_ty, &data.inputs);
                if let FnRetTy::Ty(ty) = &data.output {
                    // `-> Foo` syntax is essentially an associated type binding,
                    // so it is also allowed to contain nested `impl Trait`.
                    self.with_impl_trait(None, |this| this.visit_ty(ty));
                }
            }
        }
    }

    fn visit_generics(&mut self, generics: &'a Generics) {
        let cg_defaults = self.session.features_untracked().unordered_const_ty_params();

        let mut prev_param_default = None;
        for param in &generics.params {
            match param.kind {
                GenericParamKind::Lifetime => (),
                GenericParamKind::Type { default: Some(_), .. }
                | GenericParamKind::Const { default: Some(_), .. } => {
                    prev_param_default = Some(param.ident.span);
                }
                GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                    if let Some(span) = prev_param_default {
                        let mut err = self.err_handler().struct_span_err(
                            span,
                            "generic parameters with a default must be trailing",
                        );
                        if matches!(param.kind, GenericParamKind::Const { .. }) && !cg_defaults {
                            err.note(
                                "using type defaults and const parameters \
                                 in the same parameter list is currently not permitted",
                            );
                        }
                        err.emit();
                        break;
                    }
                }
            }
        }

        validate_generic_param_order(
            self.session,
            self.err_handler(),
            &generics.params,
            generics.span,
        );

        for predicate in &generics.where_clause.predicates {
            if let WherePredicate::EqPredicate(ref predicate) = *predicate {
                deny_equality_constraints(self, predicate, generics);
            }
        }
        walk_list!(self, visit_generic_param, &generics.params);
        for predicate in &generics.where_clause.predicates {
            match predicate {
                WherePredicate::BoundPredicate(bound_pred) => {
                    // A type binding, eg `for<'c> Foo: Send+Clone+'c`
                    self.check_late_bound_lifetime_defs(&bound_pred.bound_generic_params);

                    // This is slightly complicated. Our representation for poly-trait-refs contains a single
                    // binder and thus we only allow a single level of quantification. However,
                    // the syntax of Rust permits quantification in two places in where clauses,
                    // e.g., `T: for <'a> Foo<'a>` and `for <'a, 'b> &'b T: Foo<'a>`. If both are
                    // defined, then error.
                    if !bound_pred.bound_generic_params.is_empty() {
                        for bound in &bound_pred.bounds {
                            match bound {
                                GenericBound::Trait(t, _) => {
                                    if !t.bound_generic_params.is_empty() {
                                        struct_span_err!(
                                            self.err_handler(),
                                            t.span,
                                            E0316,
                                            "nested quantification of lifetimes"
                                        )
                                        .emit();
                                    }
                                }
                                GenericBound::Outlives(_) => {}
                            }
                        }
                    }
                }
                _ => {}
            }
            self.visit_where_predicate(predicate);
        }
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
                if !self.is_tilde_const_allowed {
                    self.err_handler()
                        .struct_span_err(bound.span(), "`~const` is not allowed here")
                        .note("only allowed on bounds on traits' associated types and functions, const fns, const impls and its associated functions")
                        .emit();
                }
            }

            GenericBound::Trait(_, TraitBoundModifier::MaybeConstMaybe) => {
                self.err_handler()
                    .span_err(bound.span(), "`~const` and `?` are mutually exclusive");
            }

            _ => {}
        }

        visit::walk_param_bound(self, bound)
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match &pat.kind {
            PatKind::Lit(expr) => {
                self.check_expr_within_pat(expr, false);
            }
            PatKind::Range(start, end, _) => {
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

        // Functions cannot both be `const async`
        if let Some(FnHeader {
            constness: Const::Yes(cspan),
            asyncness: Async::Yes { span: aspan, .. },
            ..
        }) = fk.header()
        {
            self.err_handler()
                .struct_span_err(
                    vec![*cspan, *aspan],
                    "functions cannot be both `const` and `async`",
                )
                .span_label(*cspan, "`const` because of this")
                .span_label(*aspan, "`async` because of this")
                .span_label(span, "") // Point at the fn header.
                .emit();
        }

        if let FnKind::Fn(
            _,
            _,
            FnSig { span: sig_span, header: FnHeader { ext: Extern::Implicit, .. }, .. },
            _,
            _,
        ) = fk
        {
            self.maybe_lint_missing_abi(*sig_span, id);
        }

        // Functions without bodies cannot have patterns.
        if let FnKind::Fn(ctxt, _, sig, _, None) = fk {
            Self::check_decl_no_pat(&sig.decl, |span, ident, mut_ident| {
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
                    if let Some(ident) = ident {
                        let diag = BuiltinLintDiagnostics::PatternsInFnsWithoutBody(span, ident);
                        self.lint_buffer.buffer_lint_with_diagnostic(
                            PATTERNS_IN_FNS_WITHOUT_BODY,
                            id,
                            span,
                            msg,
                            diag,
                        )
                    }
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
        if self.session.contains_name(&item.attrs, sym::no_mangle) {
            self.check_nomangle_item_asciionly(item.ident, item.span);
        }

        if ctxt == AssocCtxt::Trait || !self.in_trait_impl {
            self.check_defaultness(item.span, item.kind.defaultness());
        }

        if ctxt == AssocCtxt::Impl {
            match &item.kind {
                AssocItemKind::Const(_, _, body) => {
                    self.check_impl_item_provided(item.span, body, "constant", " = <expr>;");
                }
                AssocItemKind::Fn(box Fn { body, .. }) => {
                    self.check_impl_item_provided(item.span, body, "function", " { <body> }");
                }
                AssocItemKind::TyAlias(box TyAlias { bounds, ty, .. }) => {
                    self.check_impl_item_provided(item.span, ty, "type", " = <type>;");
                    self.check_type_no_bounds(bounds, "`impl`s");
                }
                _ => {}
            }
        }

        if ctxt == AssocCtxt::Trait || self.in_trait_impl {
            self.invalid_visibility(&item.vis, None);
            if let AssocItemKind::Fn(box Fn { sig, .. }) = &item.kind {
                self.check_trait_fn_not_const(sig.header.constness);
                self.check_trait_fn_not_async(item.span, sig.header.asyncness);
            }
        }

        if let AssocItemKind::Const(..) = item.kind {
            self.check_item_named(item.ident, "const");
        }

        match item.kind {
            AssocItemKind::TyAlias(box TyAlias { ref generics, ref bounds, ref ty, .. })
                if ctxt == AssocCtxt::Trait =>
            {
                self.visit_vis(&item.vis);
                self.visit_ident(item.ident);
                walk_list!(self, visit_attribute, &item.attrs);
                self.with_tilde_const_allowed(|this| {
                    this.visit_generics(generics);
                    walk_list!(this, visit_param_bound, bounds);
                });
                walk_list!(self, visit_ty, ty);
            }
            AssocItemKind::Fn(box Fn { ref sig, ref generics, ref body, .. })
                if self.in_const_trait_impl
                    || ctxt == AssocCtxt::Trait
                    || matches!(sig.header.constness, Const::Yes(_)) =>
            {
                self.visit_vis(&item.vis);
                self.visit_ident(item.ident);
                self.with_tilde_const_allowed(|this| this.visit_generics(generics));
                let kind =
                    FnKind::Fn(FnCtxt::Assoc(ctxt), item.ident, sig, &item.vis, body.as_deref());
                self.visit_fn(kind, item.span, item.id);
            }
            _ => self
                .with_in_trait_impl(false, None, |this| visit::walk_assoc_item(this, item, ctxt)),
        }
    }
}

/// When encountering an equality constraint in a `where` clause, emit an error. If the code seems
/// like it's setting an associated type, provide an appropriate suggestion.
fn deny_equality_constraints(
    this: &mut AstValidator<'_>,
    predicate: &WhereEqPredicate,
    generics: &Generics,
) {
    let mut err = this.err_handler().struct_span_err(
        predicate.span,
        "equality constraints are not yet supported in `where` clauses",
    );
    err.span_label(predicate.span, "not supported");

    // Given `<A as Foo>::Bar = RhsTy`, suggest `A: Foo<Bar = RhsTy>`.
    if let TyKind::Path(Some(qself), full_path) = &predicate.lhs_ty.kind {
        if let TyKind::Path(None, path) = &qself.ty.kind {
            match &path.segments[..] {
                [PathSegment { ident, args: None, .. }] => {
                    for param in &generics.params {
                        if param.ident == *ident {
                            let param = ident;
                            match &full_path.segments[qself.position..] {
                                [PathSegment { ident, args, .. }] => {
                                    // Make a new `Path` from `foo::Bar` to `Foo<Bar = RhsTy>`.
                                    let mut assoc_path = full_path.clone();
                                    // Remove `Bar` from `Foo::Bar`.
                                    assoc_path.segments.pop();
                                    let len = assoc_path.segments.len() - 1;
                                    let gen_args = args.as_ref().map(|p| (**p).clone());
                                    // Build `<Bar = RhsTy>`.
                                    let arg = AngleBracketedArg::Constraint(AssocTyConstraint {
                                        id: rustc_ast::node_id::DUMMY_NODE_ID,
                                        ident: *ident,
                                        gen_args,
                                        kind: AssocTyConstraintKind::Equality {
                                            ty: predicate.rhs_ty.clone(),
                                        },
                                        span: ident.span,
                                    });
                                    // Add `<Bar = RhsTy>` to `Foo`.
                                    match &mut assoc_path.segments[len].args {
                                        Some(args) => match args.deref_mut() {
                                            GenericArgs::Parenthesized(_) => continue,
                                            GenericArgs::AngleBracketed(args) => {
                                                args.args.push(arg);
                                            }
                                        },
                                        empty_args => {
                                            *empty_args = AngleBracketedArgs {
                                                span: ident.span,
                                                args: vec![arg],
                                            }
                                            .into();
                                        }
                                    }
                                    err.span_suggestion_verbose(
                                        predicate.span,
                                        &format!(
                                            "if `{}` is an associated type you're trying to set, \
                                            use the associated type binding syntax",
                                            ident
                                        ),
                                        format!(
                                            "{}: {}",
                                            param,
                                            pprust::path_to_string(&assoc_path)
                                        ),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                                _ => {}
                            };
                        }
                    }
                }
                _ => {}
            }
        }
    }
    err.note(
        "see issue #20041 <https://github.com/rust-lang/rust/issues/20041> for more information",
    );
    err.emit();
}

pub fn check_crate(session: &Session, krate: &Crate, lints: &mut LintBuffer) -> bool {
    let mut validator = AstValidator {
        session,
        extern_mod: None,
        in_trait_impl: false,
        in_const_trait_impl: false,
        has_proc_macro_decls: false,
        outer_impl_trait: None,
        is_tilde_const_allowed: false,
        is_impl_trait_banned: false,
        is_assoc_ty_bound_banned: false,
        is_let_allowed: false,
        lint_buffer: lints,
    };
    visit::walk_crate(&mut validator, krate);

    validator.has_proc_macro_decls
}
