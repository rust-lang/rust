// Validate AST before lowering it to HIR.
//
// This pass is supposed to catch things that fit into AST data structures,
// but not permitted by the language. It runs after expansion when AST is frozen,
// so it can check for erroneous constructions produced by syntax extensions.
// This pass is supposed to perform only simple checks not requiring name resolution
// or type checking or some other kind of complex analysis.

use std::mem;
use syntax::print::pprust;
use rustc::lint;
use rustc::lint::builtin::{BuiltinLintDiagnostics, NESTED_IMPL_TRAIT};
use rustc::session::Session;
use rustc_data_structures::fx::FxHashMap;
use syntax::ast::*;
use syntax::attr;
use syntax::feature_gate::is_builtin_attr;
use syntax::source_map::Spanned;
use syntax::symbol::{kw, sym};
use syntax::visit::{self, Visitor};
use syntax::{span_err, struct_span_err, walk_list};
use syntax_ext::proc_macro_decls::is_proc_macro_attr;
use syntax_pos::{Span, MultiSpan};
use errors::{Applicability, FatalError};

#[derive(Copy, Clone, Debug)]
struct OuterImplTrait {
    span: Span,

    /// rust-lang/rust#57979: a bug in original implementation caused
    /// us to fail sometimes to record an outer `impl Trait`.
    /// Therefore, in order to reliably issue a warning (rather than
    /// an error) in the *precise* places where we are newly injecting
    /// the diagnostic, we have to distinguish between the places
    /// where the outer `impl Trait` has always been recorded, versus
    /// the places where it has only recently started being recorded.
    only_recorded_since_pull_request_57730: bool,
}

impl OuterImplTrait {
    /// This controls whether we should downgrade the nested impl
    /// trait diagnostic to a warning rather than an error, based on
    /// whether the outer impl trait had been improperly skipped in
    /// earlier implementations of the analysis on the stable
    /// compiler.
    fn should_warn_instead_of_error(&self) -> bool {
        self.only_recorded_since_pull_request_57730
    }
}

struct AstValidator<'a> {
    session: &'a Session,
    has_proc_macro_decls: bool,
    has_global_allocator: bool,

    /// Used to ban nested `impl Trait`, e.g., `impl Into<impl Debug>`.
    /// Nested `impl Trait` _is_ allowed in associated type position,
    /// e.g., `impl Iterator<Item = impl Debug>`.
    outer_impl_trait: Option<OuterImplTrait>,

    /// Used to ban `impl Trait` in path projections like `<impl Iterator>::Item`
    /// or `Foo::Bar<impl Trait>`
    is_impl_trait_banned: bool,

    /// Used to ban associated type bounds (i.e., `Type<AssocType: Bounds>`) in
    /// certain positions.
    is_assoc_ty_bound_banned: bool,

    /// rust-lang/rust#57979: the ban of nested `impl Trait` was buggy
    /// until PRs #57730 and #57981 landed: it would jump directly to
    /// walk_ty rather than visit_ty (or skip recurring entirely for
    /// impl trait in projections), and thus miss some cases. We track
    /// whether we should downgrade to a warning for short-term via
    /// these booleans.
    warning_period_57979_didnt_record_next_impl_trait: bool,
    warning_period_57979_impl_trait_in_proj: bool,
}

impl<'a> AstValidator<'a> {
    fn with_impl_trait_in_proj_warning<T>(&mut self, v: bool, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = mem::replace(&mut self.warning_period_57979_impl_trait_in_proj, v);
        let ret = f(self);
        self.warning_period_57979_impl_trait_in_proj = old;
        ret
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

    fn with_impl_trait(&mut self, outer: Option<OuterImplTrait>, f: impl FnOnce(&mut Self)) {
        let old = mem::replace(&mut self.outer_impl_trait, outer);
        f(self);
        self.outer_impl_trait = old;
    }

    fn visit_assoc_ty_constraint_from_generic_args(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Equality { ref ty } => {
                // rust-lang/rust#57979: bug in old `visit_generic_args` called
                // `walk_ty` rather than `visit_ty`, skipping outer `impl Trait`
                // if it happened to occur at `ty`.
                if let TyKind::ImplTrait(..) = ty.node {
                    self.warning_period_57979_didnt_record_next_impl_trait = true;
                }
            }
            AssocTyConstraintKind::Bound { .. } => {
                if self.is_assoc_ty_bound_banned {
                    self.err_handler().span_err(constraint.span,
                        "associated type bounds are not allowed within structs, enums, or unions"
                    );
                }
            }
        }
        self.visit_assoc_ty_constraint(constraint);
    }

    fn visit_ty_from_generic_args(&mut self, ty: &'a Ty) {
        // rust-lang/rust#57979: bug in old `visit_generic_args` called
        // `walk_ty` rather than `visit_ty`, skippping outer `impl Trait`
        // if it happened to occur at `ty`.
        if let TyKind::ImplTrait(..) = ty.node {
            self.warning_period_57979_didnt_record_next_impl_trait = true;
        }
        self.visit_ty(ty);
    }

    fn outer_impl_trait(&mut self, span: Span) -> OuterImplTrait {
        let only_recorded_since_pull_request_57730 =
            self.warning_period_57979_didnt_record_next_impl_trait;

        // (This flag is designed to be set to `true`, and then only
        // reach the construction point for the outer impl trait once,
        // so its safe and easiest to unconditionally reset it to
        // false.)
        self.warning_period_57979_didnt_record_next_impl_trait = false;

        OuterImplTrait {
            span, only_recorded_since_pull_request_57730,
        }
    }

    // Mirrors `visit::walk_ty`, but tracks relevant state.
    fn walk_ty(&mut self, t: &'a Ty) {
        match t.node {
            TyKind::ImplTrait(..) => {
                let outer_impl_trait = self.outer_impl_trait(t.span);
                self.with_impl_trait(Some(outer_impl_trait), |this| visit::walk_ty(this, t))
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

    fn err_handler(&self) -> &errors::Handler {
        &self.session.diagnostic()
    }

    fn check_lifetime(&self, ident: Ident) {
        let valid_names = [kw::UnderscoreLifetime,
                           kw::StaticLifetime,
                           kw::Invalid];
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
            return
        }

        let mut err = struct_span_err!(self.session,
                                        vis.span,
                                        E0449,
                                        "unnecessary visibility qualifier");
        if vis.node.is_pub() {
            err.span_label(vis.span, "`pub` not permitted here because it's implied");
        }
        if let Some(note) = note {
            err.note(note);
        }
        err.emit();
    }

    fn check_decl_no_pat<ReportFn: Fn(Span, bool)>(&self, decl: &FnDecl, report_err: ReportFn) {
        for arg in &decl.inputs {
            match arg.pat.node {
                PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), _, None) |
                PatKind::Wild => {}
                PatKind::Ident(BindingMode::ByValue(Mutability::Mutable), _, None) =>
                    report_err(arg.pat.span, true),
                _ => report_err(arg.pat.span, false),
            }
        }
    }

    fn check_trait_fn_not_async(&self, span: Span, asyncness: IsAsync) {
        if asyncness.is_async() {
            struct_span_err!(self.session, span, E0706,
                             "trait fns cannot be declared `async`").emit()
        }
    }

    fn check_trait_fn_not_const(&self, constness: Spanned<Constness>) {
        if constness.node == Constness::Const {
            struct_span_err!(self.session, constness.span, E0379,
                             "trait fns cannot be declared const")
                .span_label(constness.span, "trait fns cannot be const")
                .emit();
        }
    }

    fn no_questions_in_bounds(&self, bounds: &GenericBounds, where_: &str, is_trait: bool) {
        for bound in bounds {
            if let GenericBound::Trait(ref poly, TraitBoundModifier::Maybe) = *bound {
                let mut err = self.err_handler().struct_span_err(poly.span,
                    &format!("`?Trait` is not permitted in {}", where_));
                if is_trait {
                    err.note(&format!("traits are `?{}` by default", poly.trait_ref.path));
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
        match expr.node {
            ExprKind::Lit(..) => {}
            ExprKind::Path(..) if allow_paths => {}
            ExprKind::Unary(UnOp::Neg, ref inner)
                if match inner.node { ExprKind::Lit(_) => true, _ => false } => {}
            _ => self.err_handler().span_err(expr.span, "arbitrary expressions aren't allowed \
                                                         in patterns")
        }
    }

    fn check_late_bound_lifetime_defs(&self, params: &[GenericParam]) {
        // Check only lifetime parameters are present and that the lifetime
        // parameters that are present have no bounds.
        let non_lt_param_spans: Vec<_> = params.iter().filter_map(|param| match param.kind {
            GenericParamKind::Lifetime { .. } => {
                if !param.bounds.is_empty() {
                    let spans: Vec<_> = param.bounds.iter().map(|b| b.span()).collect();
                    self.err_handler()
                        .span_err(spans, "lifetime bounds cannot be used in this context");
                }
                None
            }
            _ => Some(param.ident.span),
        }).collect();
        if !non_lt_param_spans.is_empty() {
            self.err_handler().span_err(non_lt_param_spans,
                "only lifetime parameters can be used in this context");
        }
    }

    fn check_fn_decl(&self, fn_decl: &FnDecl) {
        fn_decl
            .inputs
            .iter()
            .flat_map(|i| i.attrs.as_ref())
            .filter(|attr| {
                let arr = [sym::allow, sym::cfg, sym::cfg_attr, sym::deny, sym::forbid, sym::warn];
                !arr.contains(&attr.name_or_empty()) && is_builtin_attr(attr)
            })
            .for_each(|attr| if attr.is_sugared_doc {
                let mut err = self.err_handler().struct_span_err(
                    attr.span,
                    "documentation comments cannot be applied to function parameters"
                );
                err.span_label(attr.span, "doc comments are not allowed here");
                err.emit();
            }
            else {
                self.err_handler().span_err(attr.span, "allow, cfg, cfg_attr, deny, \
                forbid, and warn are the only allowed built-in attributes in function parameters")
            });
    }
}

enum GenericPosition {
    Param,
    Arg,
}

fn validate_generics_order<'a>(
    sess: &Session,
    handler: &errors::Handler,
    generics: impl Iterator<
        Item = (
            ParamKindOrd,
            Option<&'a [GenericBound]>,
            Span,
            Option<String>
        ),
    >,
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
        let mut err = handler.struct_span_err(spans.clone(),
            &format!(
                "{} {pos}s must be declared prior to {} {pos}s",
                param_ord,
                max_param,
                pos = pos_str,
            ));
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
    fn visit_expr(&mut self, expr: &'a Expr) {
        match &expr.node {
            ExprKind::Closure(_, _, _, fn_decl, _, _) => {
                self.check_fn_decl(fn_decl);
            }
            ExprKind::InlineAsm(..) if !self.session.target.target.options.allow_asm => {
                span_err!(self.session, expr.span, E0472, "asm! is unsupported on this target");
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            TyKind::BareFn(ref bfty) => {
                self.check_fn_decl(&bfty.decl);
                self.check_decl_no_pat(&bfty.decl, |span, _| {
                    struct_span_err!(self.session, span, E0561,
                                     "patterns aren't allowed in function pointer types").emit();
                });
                self.check_late_bound_lifetime_defs(&bfty.generic_params);
            }
            TyKind::TraitObject(ref bounds, ..) => {
                let mut any_lifetime_bounds = false;
                for bound in bounds {
                    if let GenericBound::Outlives(ref lifetime) = *bound {
                        if any_lifetime_bounds {
                            span_err!(self.session, lifetime.ident.span, E0226,
                                      "only a single explicit lifetime bound is permitted");
                            break;
                        }
                        any_lifetime_bounds = true;
                    }
                }
                self.no_questions_in_bounds(bounds, "trait object types", false);
            }
            TyKind::ImplTrait(_, ref bounds) => {
                if self.is_impl_trait_banned {
                    if self.warning_period_57979_impl_trait_in_proj {
                        self.session.buffer_lint(
                            NESTED_IMPL_TRAIT, ty.id, ty.span,
                            "`impl Trait` is not allowed in path parameters");
                    } else {
                        struct_span_err!(self.session, ty.span, E0667,
                            "`impl Trait` is not allowed in path parameters").emit();
                    }
                }

                if let Some(outer_impl_trait) = self.outer_impl_trait {
                    if outer_impl_trait.should_warn_instead_of_error() {
                        self.session.buffer_lint_with_diagnostic(
                            NESTED_IMPL_TRAIT, ty.id, ty.span,
                            "nested `impl Trait` is not allowed",
                            BuiltinLintDiagnostics::NestedImplTrait {
                                outer_impl_trait_span: outer_impl_trait.span,
                                inner_impl_trait_span: ty.span,
                            });
                    } else {
                        struct_span_err!(self.session, ty.span, E0666,
                            "nested `impl Trait` is not allowed")
                            .span_label(outer_impl_trait.span, "outer `impl Trait`")
                            .span_label(ty.span, "nested `impl Trait` here")
                            .emit();
                    }
                }

                if !bounds.iter()
                          .any(|b| if let GenericBound::Trait(..) = *b { true } else { false }) {
                    self.err_handler().span_err(ty.span, "at least one trait must be specified");
                }

                self.with_impl_trait_in_proj_warning(true, |this| this.walk_ty(ty));
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
        if item.attrs.iter().any(|attr| is_proc_macro_attr(attr)  ) {
            self.has_proc_macro_decls = true;
        }

        if attr::contains_name(&item.attrs, sym::global_allocator) {
            self.has_global_allocator = true;
        }

        match item.node {
            ItemKind::Impl(unsafety, polarity, _, _, Some(..), ref ty, ref impl_items) => {
                self.invalid_visibility(&item.vis, None);
                if let TyKind::Err = ty.node {
                    self.err_handler()
                        .struct_span_err(item.span, "`impl Trait for .. {}` is an obsolete syntax")
                        .help("use `auto trait Trait {}` instead").emit();
                }
                if unsafety == Unsafety::Unsafe && polarity == ImplPolarity::Negative {
                    span_err!(self.session, item.span, E0198, "negative impls cannot be unsafe");
                }
                for impl_item in impl_items {
                    self.invalid_visibility(&impl_item.vis, None);
                    if let ImplItemKind::Method(ref sig, _) = impl_item.node {
                        self.check_trait_fn_not_const(sig.header.constness);
                        self.check_trait_fn_not_async(impl_item.span, sig.header.asyncness.node);
                    }
                }
            }
            ItemKind::Impl(unsafety, polarity, defaultness, _, None, _, _) => {
                self.invalid_visibility(&item.vis,
                                        Some("place qualifiers on individual impl items instead"));
                if unsafety == Unsafety::Unsafe {
                    span_err!(self.session, item.span, E0197, "inherent impls cannot be unsafe");
                }
                if polarity == ImplPolarity::Negative {
                    self.err_handler().span_err(item.span, "inherent impls cannot be negative");
                }
                if defaultness == Defaultness::Default {
                    self.err_handler()
                        .struct_span_err(item.span, "inherent impls cannot be default")
                        .note("only trait implementations may be annotated with default").emit();
                }
            }
            ItemKind::Fn(ref decl, ref header, ref generics, _) => {
                self.visit_fn_header(header);
                self.check_fn_decl(decl);
                // We currently do not permit const generics in `const fn`, as
                // this is tantamount to allowing compile-time dependent typing.
                if header.constness.node == Constness::Const {
                    // Look for const generics and error if we find any.
                    for param in &generics.params {
                        match param.kind {
                            GenericParamKind::Const { .. } => {
                                self.err_handler()
                                    .struct_span_err(
                                        item.span,
                                        "const parameters are not permitted in `const fn`",
                                    )
                                    .emit();
                            }
                            _ => {}
                        }
                    }
                }
            }
            ItemKind::ForeignMod(..) => {
                self.invalid_visibility(
                    &item.vis,
                    Some("place qualifiers on individual foreign items instead"),
                );
            }
            ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.invalid_visibility(&field.vis, None);
                    }
                }
            }
            ItemKind::Trait(is_auto, _, ref generics, ref bounds, ref trait_items) => {
                if is_auto == IsAuto::Yes {
                    // Auto traits cannot have generics, super traits nor contain items.
                    if !generics.params.is_empty() {
                        struct_span_err!(self.session, item.span, E0567,
                            "auto traits cannot have generic parameters"
                        ).emit();
                    }
                    if !bounds.is_empty() {
                        struct_span_err!(self.session, item.span, E0568,
                            "auto traits cannot have super traits"
                        ).emit();
                    }
                    if !trait_items.is_empty() {
                        struct_span_err!(self.session, item.span, E0380,
                            "auto traits cannot have methods or associated items"
                        ).emit();
                    }
                }
                self.no_questions_in_bounds(bounds, "supertraits", true);
                for trait_item in trait_items {
                    if let TraitItemKind::Method(ref sig, ref block) = trait_item.node {
                        self.check_fn_decl(&sig.decl);
                        self.check_trait_fn_not_async(trait_item.span, sig.header.asyncness.node);
                        self.check_trait_fn_not_const(sig.header.constness);
                        if block.is_none() {
                            self.check_decl_no_pat(&sig.decl, |span, mut_ident| {
                                if mut_ident {
                                    self.session.buffer_lint(
                                        lint::builtin::PATTERNS_IN_FNS_WITHOUT_BODY,
                                        trait_item.id, span,
                                        "patterns aren't allowed in methods without bodies");
                                } else {
                                    struct_span_err!(self.session, span, E0642,
                                        "patterns aren't allowed in methods without bodies").emit();
                                }
                            });
                        }
                    }
                }
            }
            ItemKind::Mod(_) => {
                // Ensure that `path` attributes on modules are recorded as used (cf. issue #35584).
                attr::first_attr_value_str_by_name(&item.attrs, sym::path);
                if attr::contains_name(&item.attrs, sym::warn_directory_ownership) {
                    let lint = lint::builtin::LEGACY_DIRECTORY_OWNERSHIP;
                    let msg = "cannot declare a new module at this location";
                    self.session.buffer_lint(lint, item.id, item.span, msg);
                }
            }
            ItemKind::Union(ref vdata, _) => {
                if let VariantData::Tuple(..) | VariantData::Unit(..) = vdata {
                    self.err_handler().span_err(item.span,
                                                "tuple and unit unions are not permitted");
                }
                if vdata.fields().is_empty() {
                    self.err_handler().span_err(item.span,
                                                "unions cannot have zero fields");
                }
            }
            ItemKind::Existential(ref bounds, _) => {
                if !bounds.iter()
                          .any(|b| if let GenericBound::Trait(..) = *b { true } else { false }) {
                    let msp = MultiSpan::from_spans(bounds.iter()
                        .map(|bound| bound.span()).collect());
                    self.err_handler().span_err(msp, "at least one trait must be specified");
                }
            }
            _ => {}
        }

        visit::walk_item(self, item)
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        match fi.node {
            ForeignItemKind::Fn(ref decl, _) => {
                self.check_fn_decl(decl);
                self.check_decl_no_pat(decl, |span, _| {
                    struct_span_err!(self.session, span, E0130,
                                     "patterns aren't allowed in foreign function declarations")
                        .span_label(span, "pattern not allowed in foreign function").emit();
                });
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
                        (match arg {
                            GenericArg::Lifetime(..) => ParamKindOrd::Lifetime,
                            GenericArg::Type(..) => ParamKindOrd::Type,
                            GenericArg::Const(..) => ParamKindOrd::Const,
                        }, None, arg.span(), None)
                    }),
                    GenericPosition::Arg,
                    generic_args.span(),
                );

                // Type bindings such as `Item = impl Debug` in `Iterator<Item = Debug>`
                // are allowed to contain nested `impl Trait`.
                self.with_impl_trait(None, |this| {
                    walk_list!(this, visit_assoc_ty_constraint_from_generic_args,
                        &data.constraints);
                });
            }
            GenericArgs::Parenthesized(ref data) => {
                walk_list!(self, visit_ty, &data.inputs);
                if let Some(ref type_) = data.output {
                    // `-> Foo` syntax is essentially an associated type binding,
                    // so it is also allowed to contain nested `impl Trait`.
                    self.with_impl_trait(None, |this| this.visit_ty_from_generic_args(type_));
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
                    .span_err(predicate.span, "equality constraints are not yet \
                                               supported in where clauses (see #20041)");
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

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.node {
            PatKind::Lit(ref expr) => {
                self.check_expr_within_pat(expr, false);
            }
            PatKind::Range(ref start, ref end, _) => {
                self.check_expr_within_pat(start, true);
                self.check_expr_within_pat(end, true);
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

    fn visit_variant_data(&mut self, s: &'a VariantData, _: Ident,
                          _: &'a Generics, _: NodeId, _: Span) {
        self.with_banned_assoc_ty_bound(|this| visit::walk_struct_def(this, s))
    }

    fn visit_enum_def(&mut self, enum_definition: &'a EnumDef,
                      generics: &'a Generics, item_id: NodeId, _: Span) {
        self.with_banned_assoc_ty_bound(
            |this| visit::walk_enum_def(this, enum_definition, generics, item_id))
    }

    fn visit_mac(&mut self, mac: &Spanned<Mac_>) {
        // when a new macro kind is added but the author forgets to set it up for expansion
        // because that's the only part that won't cause a compiler error
        self.session.diagnostic()
            .span_bug(mac.span, "macro invocation missed in expansion; did you forget to override \
                                 the relevant `fold_*()` method in `PlaceholderExpander`?");
    }

    fn visit_impl_item(&mut self, ii: &'a ImplItem) {
        match ii.node {
            ImplItemKind::Method(ref sig, _) => {
                self.check_fn_decl(&sig.decl);
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii);
    }
}

pub fn check_crate(session: &Session, krate: &Crate) -> (bool, bool) {
    let mut validator = AstValidator {
        session,
        has_proc_macro_decls: false,
        has_global_allocator: false,
        outer_impl_trait: None,
        is_impl_trait_banned: false,
        is_assoc_ty_bound_banned: false,
        warning_period_57979_didnt_record_next_impl_trait: false,
        warning_period_57979_impl_trait_in_proj: false,
    };
    visit::walk_crate(&mut validator, krate);

    (validator.has_proc_macro_decls, validator.has_global_allocator)
}
