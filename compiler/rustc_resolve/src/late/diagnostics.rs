use crate::diagnostics::{ImportSuggestion, LabelSuggestion, TypoSuggestion};
use crate::late::lifetimes::{ElisionFailureInfo, LifetimeContext};
use crate::late::{AliasPossibility, LateResolutionVisitor, RibKind};
use crate::path_names_to_string;
use crate::{CrateLint, Module, ModuleKind, ModuleOrUniformRoot};
use crate::{PathResult, PathSource, Segment};

use rustc_ast::visit::FnKind;
use rustc_ast::{self as ast, Expr, ExprKind, Item, ItemKind, NodeId, Path, Ty, TyKind};
use rustc_ast_pretty::pprust::path_segment_to_string;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::PrimTy;
use rustc_session::parse::feature_err;
use rustc_span::hygiene::MacroKind;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, MultiSpan, Span, DUMMY_SP};

use tracing::debug;

type Res = def::Res<ast::NodeId>;

/// A field or associated item from self type suggested in case of resolution failure.
enum AssocSuggestion {
    Field,
    MethodWithSelf,
    AssocFn,
    AssocType,
    AssocConst,
}

impl AssocSuggestion {
    fn action(&self) -> &'static str {
        match self {
            AssocSuggestion::Field => "use the available field",
            AssocSuggestion::MethodWithSelf => "call the method with the fully-qualified path",
            AssocSuggestion::AssocFn => "call the associated function",
            AssocSuggestion::AssocConst => "use the associated `const`",
            AssocSuggestion::AssocType => "use the associated type",
        }
    }
}

crate enum MissingLifetimeSpot<'tcx> {
    Generics(&'tcx hir::Generics<'tcx>),
    HigherRanked { span: Span, span_type: ForLifetimeSpanType },
    Static,
}

crate enum ForLifetimeSpanType {
    BoundEmpty,
    BoundTail,
    TypeEmpty,
    TypeTail,
}

impl ForLifetimeSpanType {
    crate fn descr(&self) -> &'static str {
        match self {
            Self::BoundEmpty | Self::BoundTail => "bound",
            Self::TypeEmpty | Self::TypeTail => "type",
        }
    }

    crate fn suggestion(&self, sugg: &str) -> String {
        match self {
            Self::BoundEmpty | Self::TypeEmpty => format!("for<{}> ", sugg),
            Self::BoundTail | Self::TypeTail => format!(", {}", sugg),
        }
    }
}

impl<'tcx> Into<MissingLifetimeSpot<'tcx>> for &'tcx hir::Generics<'tcx> {
    fn into(self) -> MissingLifetimeSpot<'tcx> {
        MissingLifetimeSpot::Generics(self)
    }
}

fn is_self_type(path: &[Segment], namespace: Namespace) -> bool {
    namespace == TypeNS && path.len() == 1 && path[0].ident.name == kw::SelfUpper
}

fn is_self_value(path: &[Segment], namespace: Namespace) -> bool {
    namespace == ValueNS && path.len() == 1 && path[0].ident.name == kw::SelfLower
}

/// Gets the stringified path for an enum from an `ImportSuggestion` for an enum variant.
fn import_candidate_to_enum_paths(suggestion: &ImportSuggestion) -> (String, String) {
    let variant_path = &suggestion.path;
    let variant_path_string = path_names_to_string(variant_path);

    let path_len = suggestion.path.segments.len();
    let enum_path = ast::Path {
        span: suggestion.path.span,
        segments: suggestion.path.segments[0..path_len - 1].to_vec(),
        tokens: None,
    };
    let enum_path_string = path_names_to_string(&enum_path);

    (variant_path_string, enum_path_string)
}

impl<'a: 'ast, 'ast> LateResolutionVisitor<'a, '_, 'ast> {
    fn def_span(&self, def_id: DefId) -> Option<Span> {
        match def_id.krate {
            LOCAL_CRATE => self.r.opt_span(def_id),
            _ => Some(
                self.r
                    .session
                    .source_map()
                    .guess_head_span(self.r.cstore().get_span_untracked(def_id, self.r.session)),
            ),
        }
    }

    /// Handles error reporting for `smart_resolve_path_fragment` function.
    /// Creates base error and amends it with one short label and possibly some longer helps/notes.
    pub(crate) fn smart_resolve_report_errors(
        &mut self,
        path: &[Segment],
        span: Span,
        source: PathSource<'_>,
        res: Option<Res>,
    ) -> (DiagnosticBuilder<'a>, Vec<ImportSuggestion>) {
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);
        let is_enum_variant = &|res| matches!(res, Res::Def(DefKind::Variant, _));

        // Make the base error.
        let expected = source.descr_expected();
        let path_str = Segment::names_to_string(path);
        let item_str = path.last().unwrap().ident;
        let (base_msg, fallback_label, base_span, could_be_expr) = if let Some(res) = res {
            (
                format!("expected {}, found {} `{}`", expected, res.descr(), path_str),
                format!("not a {}", expected),
                span,
                match res {
                    Res::Def(DefKind::Fn, _) => {
                        // Verify whether this is a fn call or an Fn used as a type.
                        self.r
                            .session
                            .source_map()
                            .span_to_snippet(span)
                            .map(|snippet| snippet.ends_with(')'))
                            .unwrap_or(false)
                    }
                    Res::Def(
                        DefKind::Ctor(..) | DefKind::AssocFn | DefKind::Const | DefKind::AssocConst,
                        _,
                    )
                    | Res::SelfCtor(_)
                    | Res::PrimTy(_)
                    | Res::Local(_) => true,
                    _ => false,
                },
            )
        } else {
            let item_span = path.last().unwrap().ident.span;
            let (mod_prefix, mod_str) = if path.len() == 1 {
                (String::new(), "this scope".to_string())
            } else if path.len() == 2 && path[0].ident.name == kw::PathRoot {
                (String::new(), "the crate root".to_string())
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_prefix =
                    match self.resolve_path(mod_path, Some(TypeNS), false, span, CrateLint::No) {
                        PathResult::Module(ModuleOrUniformRoot::Module(module)) => module.res(),
                        _ => None,
                    }
                    .map_or(String::new(), |res| format!("{} ", res.descr()));
                (mod_prefix, format!("`{}`", Segment::names_to_string(mod_path)))
            };
            (
                format!("cannot find {} `{}` in {}{}", expected, item_str, mod_prefix, mod_str),
                if path_str == "async" && expected.starts_with("struct") {
                    "`async` blocks are only allowed in the 2018 edition".to_string()
                } else {
                    format!("not found in {}", mod_str)
                },
                item_span,
                false,
            )
        };

        let code = source.error_code(res.is_some());
        let mut err = self.r.session.struct_span_err_with_code(base_span, &base_msg, code);

        match (source, self.diagnostic_metadata.in_if_condition) {
            (PathSource::Expr(_), Some(Expr { span, kind: ExprKind::Assign(..), .. })) => {
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    "you might have meant to use pattern matching",
                    "let ".to_string(),
                    Applicability::MaybeIncorrect,
                );
                self.r.session.if_let_suggestions.borrow_mut().insert(*span);
            }
            _ => {}
        }

        let is_assoc_fn = self.self_type_is_available(span);
        // Emit help message for fake-self from other languages (e.g., `this` in Javascript).
        if ["this", "my"].contains(&&*item_str.as_str()) && is_assoc_fn {
            err.span_suggestion_short(
                span,
                "you might have meant to use `self` here instead",
                "self".to_string(),
                Applicability::MaybeIncorrect,
            );
            if !self.self_value_is_available(path[0].ident.span, span) {
                if let Some((FnKind::Fn(_, _, sig, ..), fn_span)) =
                    &self.diagnostic_metadata.current_function
                {
                    let (span, sugg) = if let Some(param) = sig.decl.inputs.get(0) {
                        (param.span.shrink_to_lo(), "&self, ")
                    } else {
                        (
                            self.r
                                .session
                                .source_map()
                                .span_through_char(*fn_span, '(')
                                .shrink_to_hi(),
                            "&self",
                        )
                    };
                    err.span_suggestion_verbose(
                        span,
                        "if you meant to use `self`, you are also missing a `self` receiver \
                         argument",
                        sugg.to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }

        // Emit special messages for unresolved `Self` and `self`.
        if is_self_type(path, ns) {
            err.code(rustc_errors::error_code!(E0411));
            err.span_label(
                span,
                "`Self` is only available in impls, traits, and type definitions".to_string(),
            );
            return (err, Vec::new());
        }
        if is_self_value(path, ns) {
            debug!("smart_resolve_path_fragment: E0424, source={:?}", source);

            err.code(rustc_errors::error_code!(E0424));
            err.span_label(span, match source {
                PathSource::Pat => "`self` value is a keyword and may not be bound to variables or shadowed"
                                   .to_string(),
                _ => "`self` value is a keyword only available in methods with a `self` parameter"
                     .to_string(),
            });
            if let Some((fn_kind, span)) = &self.diagnostic_metadata.current_function {
                // The current function has a `self' parameter, but we were unable to resolve
                // a reference to `self`. This can only happen if the `self` identifier we
                // are resolving came from a different hygiene context.
                if fn_kind.decl().inputs.get(0).map(|p| p.is_self()).unwrap_or(false) {
                    err.span_label(*span, "this function has a `self` parameter, but a macro invocation can only access identifiers it receives from parameters");
                } else {
                    let doesnt = if is_assoc_fn {
                        let (span, sugg) = fn_kind
                            .decl()
                            .inputs
                            .get(0)
                            .map(|p| (p.span.shrink_to_lo(), "&self, "))
                            .unwrap_or_else(|| {
                                (
                                    self.r
                                        .session
                                        .source_map()
                                        .span_through_char(*span, '(')
                                        .shrink_to_hi(),
                                    "&self",
                                )
                            });
                        err.span_suggestion_verbose(
                            span,
                            "add a `self` receiver parameter to make the associated `fn` a method",
                            sugg.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                        "doesn't"
                    } else {
                        "can't"
                    };
                    if let Some(ident) = fn_kind.ident() {
                        err.span_label(
                            ident.span,
                            &format!("this function {} have a `self` parameter", doesnt),
                        );
                    }
                }
            }
            return (err, Vec::new());
        }

        // Try to lookup name in more relaxed fashion for better error reporting.
        let ident = path.last().unwrap().ident;
        let candidates = self
            .r
            .lookup_import_candidates(ident, ns, &self.parent_scope, is_expected)
            .drain(..)
            .filter(|ImportSuggestion { did, .. }| {
                match (did, res.and_then(|res| res.opt_def_id())) {
                    (Some(suggestion_did), Some(actual_did)) => *suggestion_did != actual_did,
                    _ => true,
                }
            })
            .collect::<Vec<_>>();
        let crate_def_id = DefId::local(CRATE_DEF_INDEX);
        if candidates.is_empty() && is_expected(Res::Def(DefKind::Enum, crate_def_id)) {
            let enum_candidates =
                self.r.lookup_import_candidates(ident, ns, &self.parent_scope, is_enum_variant);

            if !enum_candidates.is_empty() {
                if let (PathSource::Type, Some(span)) =
                    (source, self.diagnostic_metadata.current_type_ascription.last())
                {
                    if self
                        .r
                        .session
                        .parse_sess
                        .type_ascription_path_suggestions
                        .borrow()
                        .contains(span)
                    {
                        // Already reported this issue on the lhs of the type ascription.
                        err.delay_as_bug();
                        return (err, candidates);
                    }
                }

                let mut enum_candidates = enum_candidates
                    .iter()
                    .map(|suggestion| import_candidate_to_enum_paths(&suggestion))
                    .collect::<Vec<_>>();
                enum_candidates.sort();

                // Contextualize for E0412 "cannot find type", but don't belabor the point
                // (that it's a variant) for E0573 "expected type, found variant".
                let preamble = if res.is_none() {
                    let others = match enum_candidates.len() {
                        1 => String::new(),
                        2 => " and 1 other".to_owned(),
                        n => format!(" and {} others", n),
                    };
                    format!("there is an enum variant `{}`{}; ", enum_candidates[0].0, others)
                } else {
                    String::new()
                };
                let msg = format!("{}try using the variant's enum", preamble);

                err.span_suggestions(
                    span,
                    &msg,
                    enum_candidates
                        .into_iter()
                        .map(|(_variant_path, enum_ty_path)| enum_ty_path)
                        // Variants re-exported in prelude doesn't mean `prelude::v1` is the
                        // type name!
                        // FIXME: is there a more principled way to do this that
                        // would work for other re-exports?
                        .filter(|enum_ty_path| enum_ty_path != "std::prelude::v1")
                        // Also write `Option` rather than `std::prelude::v1::Option`.
                        .map(|enum_ty_path| {
                            // FIXME #56861: DRY-er prelude filtering.
                            enum_ty_path.trim_start_matches("std::prelude::v1::").to_owned()
                        }),
                    Applicability::MachineApplicable,
                );
            }
        }
        if path.len() == 1 && self.self_type_is_available(span) {
            if let Some(candidate) = self.lookup_assoc_candidate(ident, ns, is_expected) {
                let self_is_available = self.self_value_is_available(path[0].ident.span, span);
                match candidate {
                    AssocSuggestion::Field => {
                        if self_is_available {
                            err.span_suggestion(
                                span,
                                "you might have meant to use the available field",
                                format!("self.{}", path_str),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_label(span, "a field by this name exists in `Self`");
                        }
                    }
                    AssocSuggestion::MethodWithSelf if self_is_available => {
                        err.span_suggestion(
                            span,
                            "you might have meant to call the method",
                            format!("self.{}", path_str),
                            Applicability::MachineApplicable,
                        );
                    }
                    AssocSuggestion::MethodWithSelf
                    | AssocSuggestion::AssocFn
                    | AssocSuggestion::AssocConst
                    | AssocSuggestion::AssocType => {
                        err.span_suggestion(
                            span,
                            &format!("you might have meant to {}", candidate.action()),
                            format!("Self::{}", path_str),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                return (err, candidates);
            }

            // If the first argument in call is `self` suggest calling a method.
            if let Some((call_span, args_span)) = self.call_has_self_arg(source) {
                let mut args_snippet = String::new();
                if let Some(args_span) = args_span {
                    if let Ok(snippet) = self.r.session.source_map().span_to_snippet(args_span) {
                        args_snippet = snippet;
                    }
                }

                err.span_suggestion(
                    call_span,
                    &format!("try calling `{}` as a method", ident),
                    format!("self.{}({})", path_str, args_snippet),
                    Applicability::MachineApplicable,
                );
                return (err, candidates);
            }
        }

        // Try Levenshtein algorithm.
        let typo_sugg = self.lookup_typo_candidate(path, ns, is_expected, span);
        // Try context-dependent help if relaxed lookup didn't work.
        if let Some(res) = res {
            if self.smart_resolve_context_dependent_help(
                &mut err,
                span,
                source,
                res,
                &path_str,
                &fallback_label,
            ) {
                // We do this to avoid losing a secondary span when we override the main error span.
                self.r.add_typo_suggestion(&mut err, typo_sugg, ident_span);
                return (err, candidates);
            }
        }

        if !self.type_ascription_suggestion(&mut err, base_span) {
            let mut fallback = false;
            if let (
                PathSource::Trait(AliasPossibility::Maybe),
                Some(Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _)),
            ) = (source, res)
            {
                if let Some(bounds @ [_, .., _]) = self.diagnostic_metadata.current_trait_object {
                    fallback = true;
                    let spans: Vec<Span> = bounds
                        .iter()
                        .map(|bound| bound.span())
                        .filter(|&sp| sp != base_span)
                        .collect();

                    let start_span = bounds.iter().map(|bound| bound.span()).next().unwrap();
                    // `end_span` is the end of the poly trait ref (Foo + 'baz + Bar><)
                    let end_span = bounds.iter().map(|bound| bound.span()).last().unwrap();
                    // `last_bound_span` is the last bound of the poly trait ref (Foo + >'baz< + Bar)
                    let last_bound_span = spans.last().cloned().unwrap();
                    let mut multi_span: MultiSpan = spans.clone().into();
                    for sp in spans {
                        let msg = if sp == last_bound_span {
                            format!(
                                "...because of {} bound{}",
                                if bounds.len() <= 2 { "this" } else { "these" },
                                if bounds.len() <= 2 { "" } else { "s" },
                            )
                        } else {
                            String::new()
                        };
                        multi_span.push_span_label(sp, msg);
                    }
                    multi_span.push_span_label(
                        base_span,
                        "expected this type to be a trait...".to_string(),
                    );
                    err.span_help(
                        multi_span,
                        "`+` is used to constrain a \"trait object\" type with lifetimes or \
                         auto-traits; structs and enums can't be bound in that way",
                    );
                    if bounds.iter().all(|bound| match bound {
                        ast::GenericBound::Outlives(_) => true,
                        ast::GenericBound::Trait(tr, _) => tr.span == base_span,
                    }) {
                        let mut sugg = vec![];
                        if base_span != start_span {
                            sugg.push((start_span.until(base_span), String::new()));
                        }
                        if base_span != end_span {
                            sugg.push((base_span.shrink_to_hi().to(end_span), String::new()));
                        }

                        err.multipart_suggestion(
                            "if you meant to use a type and not a trait here, remove the bounds",
                            sugg,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }

            fallback |= self.restrict_assoc_type_in_where_clause(span, &mut err);

            if !self.r.add_typo_suggestion(&mut err, typo_sugg, ident_span) {
                fallback = true;
                match self.diagnostic_metadata.current_let_binding {
                    Some((pat_sp, Some(ty_sp), None))
                        if ty_sp.contains(base_span) && could_be_expr =>
                    {
                        err.span_suggestion_short(
                            pat_sp.between(ty_sp),
                            "use `=` if you meant to assign",
                            " = ".to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    _ => {}
                }
            }
            if fallback {
                // Fallback label.
                err.span_label(base_span, fallback_label);
            }
        }
        if let Some(err_code) = &err.code {
            if err_code == &rustc_errors::error_code!(E0425) {
                for label_rib in &self.label_ribs {
                    for (label_ident, _) in &label_rib.bindings {
                        if format!("'{}", ident) == label_ident.to_string() {
                            let msg = "a label with a similar name exists";
                            // FIXME: consider only emitting this suggestion if a label would be valid here
                            // which is pretty much only the case for `break` expressions.
                            err.span_suggestion(
                                span,
                                &msg,
                                label_ident.name.to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }
            }
        }

        (err, candidates)
    }

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    fn restrict_assoc_type_in_where_clause(
        &mut self,
        span: Span,
        err: &mut DiagnosticBuilder<'_>,
    ) -> bool {
        // Detect that we are actually in a `where` predicate.
        let (bounded_ty, bounds, where_span) =
            if let Some(ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                bounded_ty,
                bound_generic_params,
                bounds,
                span,
            })) = self.diagnostic_metadata.current_where_predicate
            {
                if !bound_generic_params.is_empty() {
                    return false;
                }
                (bounded_ty, bounds, span)
            } else {
                return false;
            };

        // Confirm that the target is an associated type.
        let (ty, position, path) = if let ast::TyKind::Path(
            Some(ast::QSelf { ty, position, .. }),
            path,
        ) = &bounded_ty.kind
        {
            // use this to verify that ident is a type param.
            let partial_res = if let Ok(Some(partial_res)) = self.resolve_qpath_anywhere(
                bounded_ty.id,
                None,
                &Segment::from_path(path),
                Namespace::TypeNS,
                span,
                true,
                CrateLint::No,
            ) {
                partial_res
            } else {
                return false;
            };
            if !(matches!(
                partial_res.base_res(),
                hir::def::Res::Def(hir::def::DefKind::AssocTy, _)
            ) && partial_res.unresolved_segments() == 0)
            {
                return false;
            }
            (ty, position, path)
        } else {
            return false;
        };

        if let ast::TyKind::Path(None, type_param_path) = &ty.peel_refs().kind {
            // Confirm that the `SelfTy` is a type parameter.
            let partial_res = if let Ok(Some(partial_res)) = self.resolve_qpath_anywhere(
                bounded_ty.id,
                None,
                &Segment::from_path(type_param_path),
                Namespace::TypeNS,
                span,
                true,
                CrateLint::No,
            ) {
                partial_res
            } else {
                return false;
            };
            if !(matches!(
                partial_res.base_res(),
                hir::def::Res::Def(hir::def::DefKind::TyParam, _)
            ) && partial_res.unresolved_segments() == 0)
            {
                return false;
            }
            if let (
                [ast::PathSegment { ident: constrain_ident, args: None, .. }],
                [ast::GenericBound::Trait(poly_trait_ref, ast::TraitBoundModifier::None)],
            ) = (&type_param_path.segments[..], &bounds[..])
            {
                if let [ast::PathSegment { ident, args: None, .. }] =
                    &poly_trait_ref.trait_ref.path.segments[..]
                {
                    if ident.span == span {
                        err.span_suggestion_verbose(
                            *where_span,
                            &format!("constrain the associated type to `{}`", ident),
                            format!(
                                "{}: {}<{} = {}>",
                                self.r
                                    .session
                                    .source_map()
                                    .span_to_snippet(ty.span) // Account for `<&'a T as Foo>::Bar`.
                                    .unwrap_or_else(|_| constrain_ident.to_string()),
                                path.segments[..*position]
                                    .iter()
                                    .map(|segment| path_segment_to_string(segment))
                                    .collect::<Vec<_>>()
                                    .join("::"),
                                path.segments[*position..]
                                    .iter()
                                    .map(|segment| path_segment_to_string(segment))
                                    .collect::<Vec<_>>()
                                    .join("::"),
                                ident,
                            ),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Check if the source is call expression and the first argument is `self`. If true,
    /// return the span of whole call and the span for all arguments expect the first one (`self`).
    fn call_has_self_arg(&self, source: PathSource<'_>) -> Option<(Span, Option<Span>)> {
        let mut has_self_arg = None;
        if let PathSource::Expr(Some(parent)) = source {
            match &parent.kind {
                ExprKind::Call(_, args) if !args.is_empty() => {
                    let mut expr_kind = &args[0].kind;
                    loop {
                        match expr_kind {
                            ExprKind::Path(_, arg_name) if arg_name.segments.len() == 1 => {
                                if arg_name.segments[0].ident.name == kw::SelfLower {
                                    let call_span = parent.span;
                                    let tail_args_span = if args.len() > 1 {
                                        Some(Span::new(
                                            args[1].span.lo(),
                                            args.last().unwrap().span.hi(),
                                            call_span.ctxt(),
                                        ))
                                    } else {
                                        None
                                    };
                                    has_self_arg = Some((call_span, tail_args_span));
                                }
                                break;
                            }
                            ExprKind::AddrOf(_, _, expr) => expr_kind = &expr.kind,
                            _ => break,
                        }
                    }
                }
                _ => (),
            }
        };
        has_self_arg
    }

    fn followed_by_brace(&self, span: Span) -> (bool, Option<Span>) {
        // HACK(estebank): find a better way to figure out that this was a
        // parser issue where a struct literal is being used on an expression
        // where a brace being opened means a block is being started. Look
        // ahead for the next text to see if `span` is followed by a `{`.
        let sm = self.r.session.source_map();
        let mut sp = span;
        loop {
            sp = sm.next_point(sp);
            match sm.span_to_snippet(sp) {
                Ok(ref snippet) => {
                    if snippet.chars().any(|c| !c.is_whitespace()) {
                        break;
                    }
                }
                _ => break,
            }
        }
        let followed_by_brace = matches!(sm.span_to_snippet(sp), Ok(ref snippet) if snippet == "{");
        // In case this could be a struct literal that needs to be surrounded
        // by parentheses, find the appropriate span.
        let mut i = 0;
        let mut closing_brace = None;
        loop {
            sp = sm.next_point(sp);
            match sm.span_to_snippet(sp) {
                Ok(ref snippet) => {
                    if snippet == "}" {
                        closing_brace = Some(span.to(sp));
                        break;
                    }
                }
                _ => break,
            }
            i += 1;
            // The bigger the span, the more likely we're incorrect --
            // bound it to 100 chars long.
            if i > 100 {
                break;
            }
        }
        (followed_by_brace, closing_brace)
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut DiagnosticBuilder<'a>,
        span: Span,
        source: PathSource<'_>,
        res: Res,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);

        let path_sep = |err: &mut DiagnosticBuilder<'_>, expr: &Expr| match expr.kind {
            ExprKind::Field(_, ident) => {
                err.span_suggestion(
                    expr.span,
                    "use the path separator to refer to an item",
                    format!("{}::{}", path_str, ident),
                    Applicability::MaybeIncorrect,
                );
                true
            }
            ExprKind::MethodCall(ref segment, ..) => {
                let span = expr.span.with_hi(segment.ident.span.hi());
                err.span_suggestion(
                    span,
                    "use the path separator to refer to an item",
                    format!("{}::{}", path_str, segment.ident),
                    Applicability::MaybeIncorrect,
                );
                true
            }
            _ => false,
        };

        let mut bad_struct_syntax_suggestion = |def_id: DefId| {
            let (followed_by_brace, closing_brace) = self.followed_by_brace(span);

            match source {
                PathSource::Expr(Some(
                    parent @ Expr { kind: ExprKind::Field(..) | ExprKind::MethodCall(..), .. },
                )) if path_sep(err, &parent) => {}
                PathSource::Expr(
                    None
                    | Some(Expr {
                        kind:
                            ExprKind::Path(..)
                            | ExprKind::Binary(..)
                            | ExprKind::Unary(..)
                            | ExprKind::If(..)
                            | ExprKind::While(..)
                            | ExprKind::ForLoop(..)
                            | ExprKind::Match(..),
                        ..
                    }),
                ) if followed_by_brace => {
                    if let Some(sp) = closing_brace {
                        err.span_label(span, fallback_label);
                        err.multipart_suggestion(
                            "surround the struct literal with parentheses",
                            vec![
                                (sp.shrink_to_lo(), "(".to_string()),
                                (sp.shrink_to_hi(), ")".to_string()),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_label(
                            span, // Note the parentheses surrounding the suggestion below
                            format!(
                                "you might want to surround a struct literal with parentheses: \
                                 `({} {{ /* fields */ }})`?",
                                path_str
                            ),
                        );
                    }
                }
                PathSource::Expr(_) | PathSource::TupleStruct(..) | PathSource::Pat => {
                    let span = match &source {
                        PathSource::Expr(Some(Expr {
                            span, kind: ExprKind::Call(_, _), ..
                        }))
                        | PathSource::TupleStruct(span, _) => {
                            // We want the main underline to cover the suggested code as well for
                            // cleaner output.
                            err.set_span(*span);
                            *span
                        }
                        _ => span,
                    };
                    if let Some(span) = self.def_span(def_id) {
                        err.span_label(span, &format!("`{}` defined here", path_str));
                    }
                    let (tail, descr, applicability) = match source {
                        PathSource::Pat | PathSource::TupleStruct(..) => {
                            ("", "pattern", Applicability::MachineApplicable)
                        }
                        _ => (": val", "literal", Applicability::HasPlaceholders),
                    };
                    let (fields, applicability) = match self.r.field_names.get(&def_id) {
                        Some(fields) => (
                            fields
                                .iter()
                                .map(|f| format!("{}{}", f.node, tail))
                                .collect::<Vec<String>>()
                                .join(", "),
                            applicability,
                        ),
                        None => ("/* fields */".to_string(), Applicability::HasPlaceholders),
                    };
                    let pad = match self.r.field_names.get(&def_id) {
                        Some(fields) if fields.is_empty() => "",
                        _ => " ",
                    };
                    err.span_suggestion(
                        span,
                        &format!("use struct {} syntax instead", descr),
                        format!("{path_str} {{{pad}{fields}{pad}}}"),
                        applicability,
                    );
                }
                _ => {
                    err.span_label(span, fallback_label);
                }
            }
        };

        match (res, source) {
            (Res::Def(DefKind::Macro(MacroKind::Bang), _), _) => {
                err.span_label(span, fallback_label);
                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "use `!` to invoke the macro",
                    "!".to_string(),
                    Applicability::MaybeIncorrect,
                );
                if path_str == "try" && span.rust_2015() {
                    err.note("if you want the `try` keyword, you need to be in the 2018 edition");
                }
            }
            (Res::Def(DefKind::TyAlias, def_id), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if self.r.session.is_nightly_build() {
                    let msg = "you might have meant to use `#![feature(trait_alias)]` instead of a \
                               `type` alias";
                    if let Some(span) = self.def_span(def_id) {
                        err.span_help(span, msg);
                    } else {
                        err.help(msg);
                    }
                }
            }
            (Res::Def(DefKind::Mod, _), PathSource::Expr(Some(parent))) => {
                if !path_sep(err, &parent) {
                    return false;
                }
            }
            (
                Res::Def(DefKind::Enum, def_id),
                PathSource::TupleStruct(..) | PathSource::Expr(..),
            ) => {
                if self
                    .diagnostic_metadata
                    .current_type_ascription
                    .last()
                    .map(|sp| {
                        self.r
                            .session
                            .parse_sess
                            .type_ascription_path_suggestions
                            .borrow()
                            .contains(&sp)
                    })
                    .unwrap_or(false)
                {
                    err.delay_as_bug();
                    // We already suggested changing `:` into `::` during parsing.
                    return false;
                }

                self.suggest_using_enum_variant(err, source, def_id, span);
            }
            (Res::Def(DefKind::Struct, def_id), _) if ns == ValueNS => {
                let (ctor_def, ctor_vis, fields) =
                    if let Some(struct_ctor) = self.r.struct_constructors.get(&def_id).cloned() {
                        struct_ctor
                    } else {
                        bad_struct_syntax_suggestion(def_id);
                        return true;
                    };

                let is_accessible = self.r.is_accessible_from(ctor_vis, self.parent_scope.module);
                if !is_expected(ctor_def) || is_accessible {
                    return true;
                }

                let field_spans = match source {
                    // e.g. `if let Enum::TupleVariant(field1, field2) = _`
                    PathSource::TupleStruct(_, pattern_spans) => {
                        err.set_primary_message(
                            "cannot match against a tuple struct which contains private fields",
                        );

                        // Use spans of the tuple struct pattern.
                        Some(Vec::from(pattern_spans))
                    }
                    // e.g. `let _ = Enum::TupleVariant(field1, field2);`
                    _ if source.is_call() => {
                        err.set_primary_message(
                            "cannot initialize a tuple struct which contains private fields",
                        );

                        // Use spans of the tuple struct definition.
                        self.r
                            .field_names
                            .get(&def_id)
                            .map(|fields| fields.iter().map(|f| f.span).collect::<Vec<_>>())
                    }
                    _ => None,
                };

                if let Some(spans) =
                    field_spans.filter(|spans| spans.len() > 0 && fields.len() == spans.len())
                {
                    let non_visible_spans: Vec<Span> = fields
                        .iter()
                        .zip(spans.iter())
                        .filter(|(vis, _)| {
                            !self.r.is_accessible_from(**vis, self.parent_scope.module)
                        })
                        .map(|(_, span)| *span)
                        .collect();

                    if non_visible_spans.len() > 0 {
                        let mut m: rustc_span::MultiSpan = non_visible_spans.clone().into();
                        non_visible_spans
                            .into_iter()
                            .for_each(|s| m.push_span_label(s, "private field".to_string()));
                        err.span_note(m, "constructor is not visible here due to private fields");
                    }

                    return true;
                }

                err.span_label(
                    span,
                    "constructor is not visible here due to private fields".to_string(),
                );
            }
            (
                Res::Def(
                    DefKind::Union | DefKind::Variant | DefKind::Ctor(_, CtorKind::Fictive),
                    def_id,
                ),
                _,
            ) if ns == ValueNS => {
                bad_struct_syntax_suggestion(def_id);
            }
            (Res::Def(DefKind::Ctor(_, CtorKind::Fn), def_id), _) if ns == ValueNS => {
                if let Some(span) = self.def_span(def_id) {
                    err.span_label(span, &format!("`{}` defined here", path_str));
                }
                let fields =
                    self.r.field_names.get(&def_id).map_or("/* fields */".to_string(), |fields| {
                        vec!["_"; fields.len()].join(", ")
                    });
                err.span_suggestion(
                    span,
                    "use the tuple variant pattern syntax instead",
                    format!("{}({})", path_str, fields),
                    Applicability::HasPlaceholders,
                );
            }
            (Res::SelfTy(..), _) if ns == ValueNS => {
                err.span_label(span, fallback_label);
                err.note("can't use `Self` as a constructor, you must use the implemented struct");
            }
            (Res::Def(DefKind::TyAlias | DefKind::AssocTy, _), _) if ns == ValueNS => {
                err.note("can't use a type alias as a constructor");
            }
            _ => return false,
        }
        true
    }

    fn lookup_assoc_candidate<FilterFn>(
        &mut self,
        ident: Ident,
        ns: Namespace,
        filter_fn: FilterFn,
    ) -> Option<AssocSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        fn extract_node_id(t: &Ty) -> Option<NodeId> {
            match t.kind {
                TyKind::Path(None, _) => Some(t.id),
                TyKind::Rptr(_, ref mut_ty) => extract_node_id(&mut_ty.ty),
                // This doesn't handle the remaining `Ty` variants as they are not
                // that commonly the self_type, it might be interesting to provide
                // support for those in future.
                _ => None,
            }
        }

        // Fields are generally expected in the same contexts as locals.
        if filter_fn(Res::Local(ast::DUMMY_NODE_ID)) {
            if let Some(node_id) =
                self.diagnostic_metadata.current_self_type.as_ref().and_then(extract_node_id)
            {
                // Look for a field with the same name in the current self_type.
                if let Some(resolution) = self.r.partial_res_map.get(&node_id) {
                    match resolution.base_res() {
                        Res::Def(DefKind::Struct | DefKind::Union, did)
                            if resolution.unresolved_segments() == 0 =>
                        {
                            if let Some(field_names) = self.r.field_names.get(&did) {
                                if field_names
                                    .iter()
                                    .any(|&field_name| ident.name == field_name.node)
                                {
                                    return Some(AssocSuggestion::Field);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(items) = self.diagnostic_metadata.current_trait_assoc_items {
            for assoc_item in &items[..] {
                if assoc_item.ident == ident {
                    return Some(match &assoc_item.kind {
                        ast::AssocItemKind::Const(..) => AssocSuggestion::AssocConst,
                        ast::AssocItemKind::Fn(_, sig, ..) if sig.decl.has_self() => {
                            AssocSuggestion::MethodWithSelf
                        }
                        ast::AssocItemKind::Fn(..) => AssocSuggestion::AssocFn,
                        ast::AssocItemKind::TyAlias(..) => AssocSuggestion::AssocType,
                        ast::AssocItemKind::MacCall(_) => continue,
                    });
                }
            }
        }

        // Look for associated items in the current trait.
        if let Some((module, _)) = self.current_trait_ref {
            if let Ok(binding) = self.r.resolve_ident_in_module(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                &self.parent_scope,
                false,
                module.span,
            ) {
                let res = binding.res();
                if filter_fn(res) {
                    if self.r.has_self.contains(&res.def_id()) {
                        return Some(AssocSuggestion::MethodWithSelf);
                    } else {
                        match res {
                            Res::Def(DefKind::AssocFn, _) => return Some(AssocSuggestion::AssocFn),
                            Res::Def(DefKind::AssocConst, _) => {
                                return Some(AssocSuggestion::AssocConst);
                            }
                            Res::Def(DefKind::AssocTy, _) => {
                                return Some(AssocSuggestion::AssocType);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        None
    }

    fn lookup_typo_candidate(
        &mut self,
        path: &[Segment],
        ns: Namespace,
        filter_fn: &impl Fn(Res) -> bool,
        span: Span,
    ) -> Option<TypoSuggestion> {
        let mut names = Vec::new();
        if path.len() == 1 {
            // Search in lexical scope.
            // Walk backwards up the ribs in scope and collect candidates.
            for rib in self.ribs[ns].iter().rev() {
                // Locals and type parameters
                for (ident, &res) in &rib.bindings {
                    if filter_fn(res) {
                        names.push(TypoSuggestion::from_res(ident.name, res));
                    }
                }
                // Items in scope
                if let RibKind::ModuleRibKind(module) = rib.kind {
                    // Items from this module
                    self.r.add_module_candidates(module, &mut names, &filter_fn);

                    if let ModuleKind::Block(..) = module.kind {
                        // We can see through blocks
                    } else {
                        // Items from the prelude
                        if !module.no_implicit_prelude {
                            let extern_prelude = self.r.extern_prelude.clone();
                            names.extend(extern_prelude.iter().flat_map(|(ident, _)| {
                                self.r.crate_loader.maybe_process_path_extern(ident.name).and_then(
                                    |crate_id| {
                                        let crate_mod = Res::Def(
                                            DefKind::Mod,
                                            DefId { krate: crate_id, index: CRATE_DEF_INDEX },
                                        );

                                        if filter_fn(crate_mod) {
                                            Some(TypoSuggestion::from_res(ident.name, crate_mod))
                                        } else {
                                            None
                                        }
                                    },
                                )
                            }));

                            if let Some(prelude) = self.r.prelude {
                                self.r.add_module_candidates(prelude, &mut names, &filter_fn);
                            }
                        }
                        break;
                    }
                }
            }
            // Add primitive types to the mix
            if filter_fn(Res::PrimTy(PrimTy::Bool)) {
                names.extend(
                    self.r.primitive_type_table.primitive_types.iter().map(|(name, prim_ty)| {
                        TypoSuggestion::from_res(*name, Res::PrimTy(*prim_ty))
                    }),
                )
            }
        } else {
            // Search in module.
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(module) =
                self.resolve_path(mod_path, Some(TypeNS), false, span, CrateLint::No)
            {
                if let ModuleOrUniformRoot::Module(module) = module {
                    self.r.add_module_candidates(module, &mut names, &filter_fn);
                }
            }
        }

        let name = path[path.len() - 1].ident.name;
        // Make sure error reporting is deterministic.
        names.sort_by_cached_key(|suggestion| suggestion.candidate.as_str());

        match find_best_match_for_name(
            &names.iter().map(|suggestion| suggestion.candidate).collect::<Vec<Symbol>>(),
            name,
            None,
        ) {
            Some(found) if found != name => {
                names.into_iter().find(|suggestion| suggestion.candidate == found)
            }
            _ => None,
        }
    }

    /// Only used in a specific case of type ascription suggestions
    fn get_colon_suggestion_span(&self, start: Span) -> Span {
        let sm = self.r.session.source_map();
        start.to(sm.next_point(start))
    }

    fn type_ascription_suggestion(&self, err: &mut DiagnosticBuilder<'_>, base_span: Span) -> bool {
        let sm = self.r.session.source_map();
        let base_snippet = sm.span_to_snippet(base_span);
        if let Some(&sp) = self.diagnostic_metadata.current_type_ascription.last() {
            if let Ok(snippet) = sm.span_to_snippet(sp) {
                let len = snippet.trim_end().len() as u32;
                if snippet.trim() == ":" {
                    let colon_sp =
                        sp.with_lo(sp.lo() + BytePos(len - 1)).with_hi(sp.lo() + BytePos(len));
                    let mut show_label = true;
                    if sm.is_multiline(sp) {
                        err.span_suggestion_short(
                            colon_sp,
                            "maybe you meant to write `;` here",
                            ";".to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        let after_colon_sp =
                            self.get_colon_suggestion_span(colon_sp.shrink_to_hi());
                        if snippet.len() == 1 {
                            // `foo:bar`
                            err.span_suggestion(
                                colon_sp,
                                "maybe you meant to write a path separator here",
                                "::".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                            show_label = false;
                            if !self
                                .r
                                .session
                                .parse_sess
                                .type_ascription_path_suggestions
                                .borrow_mut()
                                .insert(colon_sp)
                            {
                                err.delay_as_bug();
                            }
                        }
                        if let Ok(base_snippet) = base_snippet {
                            let mut sp = after_colon_sp;
                            for _ in 0..100 {
                                // Try to find an assignment
                                sp = sm.next_point(sp);
                                let snippet = sm.span_to_snippet(sp.to(sm.next_point(sp)));
                                match snippet {
                                    Ok(ref x) if x.as_str() == "=" => {
                                        err.span_suggestion(
                                            base_span,
                                            "maybe you meant to write an assignment here",
                                            format!("let {}", base_snippet),
                                            Applicability::MaybeIncorrect,
                                        );
                                        show_label = false;
                                        break;
                                    }
                                    Ok(ref x) if x.as_str() == "\n" => break,
                                    Err(_) => break,
                                    Ok(_) => {}
                                }
                            }
                        }
                    }
                    if show_label {
                        err.span_label(
                            base_span,
                            "expecting a type here because of type ascription",
                        );
                    }
                    return show_label;
                }
            }
        }
        false
    }

    fn find_module(&mut self, def_id: DefId) -> Option<(Module<'a>, ImportSuggestion)> {
        let mut result = None;
        let mut seen_modules = FxHashSet::default();
        let mut worklist = vec![(self.r.graph_root, Vec::new())];

        while let Some((in_module, path_segments)) = worklist.pop() {
            // abort if the module is already found
            if result.is_some() {
                break;
            }

            in_module.for_each_child(self.r, |_, ident, _, name_binding| {
                // abort if the module is already found or if name_binding is private external
                if result.is_some() || !name_binding.vis.is_visible_locally() {
                    return;
                }
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));
                    let module_def_id = module.def_id().unwrap();
                    if module_def_id == def_id {
                        let path =
                            Path { span: name_binding.span, segments: path_segments, tokens: None };
                        result = Some((
                            module,
                            ImportSuggestion {
                                did: Some(def_id),
                                descr: "module",
                                path,
                                accessible: true,
                            },
                        ));
                    } else {
                        // add the module to the lookup
                        if seen_modules.insert(module_def_id) {
                            worklist.push((module, path_segments));
                        }
                    }
                }
            });
        }

        result
    }

    fn collect_enum_ctors(&mut self, def_id: DefId) -> Option<Vec<(Path, DefId, CtorKind)>> {
        self.find_module(def_id).map(|(enum_module, enum_import_suggestion)| {
            let mut variants = Vec::new();
            enum_module.for_each_child(self.r, |_, ident, _, name_binding| {
                if let Res::Def(DefKind::Ctor(CtorOf::Variant, kind), def_id) = name_binding.res() {
                    let mut segms = enum_import_suggestion.path.segments.clone();
                    segms.push(ast::PathSegment::from_ident(ident));
                    let path = Path { span: name_binding.span, segments: segms, tokens: None };
                    variants.push((path, def_id, kind));
                }
            });
            variants
        })
    }

    /// Adds a suggestion for using an enum's variant when an enum is used instead.
    fn suggest_using_enum_variant(
        &mut self,
        err: &mut DiagnosticBuilder<'a>,
        source: PathSource<'_>,
        def_id: DefId,
        span: Span,
    ) {
        let variants = match self.collect_enum_ctors(def_id) {
            Some(variants) => variants,
            None => {
                err.note("you might have meant to use one of the enum's variants");
                return;
            }
        };

        let suggest_only_tuple_variants =
            matches!(source, PathSource::TupleStruct(..)) || source.is_call();
        if suggest_only_tuple_variants {
            // Suggest only tuple variants regardless of whether they have fields and do not
            // suggest path with added parenthesis.
            let mut suggestable_variants = variants
                .iter()
                .filter(|(.., kind)| *kind == CtorKind::Fn)
                .map(|(variant, ..)| path_names_to_string(variant))
                .collect::<Vec<_>>();

            let non_suggestable_variant_count = variants.len() - suggestable_variants.len();

            let source_msg = if source.is_call() {
                "to construct"
            } else if matches!(source, PathSource::TupleStruct(..)) {
                "to match against"
            } else {
                unreachable!()
            };

            if !suggestable_variants.is_empty() {
                let msg = if non_suggestable_variant_count == 0 && suggestable_variants.len() == 1 {
                    format!("try {} the enum's variant", source_msg)
                } else {
                    format!("try {} one of the enum's variants", source_msg)
                };

                err.span_suggestions(
                    span,
                    &msg,
                    suggestable_variants.drain(..),
                    Applicability::MaybeIncorrect,
                );
            }

            // If the enum has no tuple variants..
            if non_suggestable_variant_count == variants.len() {
                err.help(&format!("the enum has no tuple variants {}", source_msg));
            }

            // If there are also non-tuple variants..
            if non_suggestable_variant_count == 1 {
                err.help(&format!(
                    "you might have meant {} the enum's non-tuple variant",
                    source_msg
                ));
            } else if non_suggestable_variant_count >= 1 {
                err.help(&format!(
                    "you might have meant {} one of the enum's non-tuple variants",
                    source_msg
                ));
            }
        } else {
            let needs_placeholder = |def_id: DefId, kind: CtorKind| {
                let has_no_fields =
                    self.r.field_names.get(&def_id).map(|f| f.is_empty()).unwrap_or(false);
                match kind {
                    CtorKind::Const => false,
                    CtorKind::Fn | CtorKind::Fictive if has_no_fields => false,
                    _ => true,
                }
            };

            let mut suggestable_variants = variants
                .iter()
                .filter(|(_, def_id, kind)| !needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .map(|(variant, kind)| match kind {
                    CtorKind::Const => variant,
                    CtorKind::Fn => format!("({}())", variant),
                    CtorKind::Fictive => format!("({} {{}})", variant),
                })
                .collect::<Vec<_>>();

            if !suggestable_variants.is_empty() {
                let msg = if suggestable_variants.len() == 1 {
                    "you might have meant to use the following enum variant"
                } else {
                    "you might have meant to use one of the following enum variants"
                };

                err.span_suggestions(
                    span,
                    msg,
                    suggestable_variants.drain(..),
                    Applicability::MaybeIncorrect,
                );
            }

            let mut suggestable_variants_with_placeholders = variants
                .iter()
                .filter(|(_, def_id, kind)| needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .filter_map(|(variant, kind)| match kind {
                    CtorKind::Fn => Some(format!("({}(/* fields */))", variant)),
                    CtorKind::Fictive => Some(format!("({} {{ /* fields */ }})", variant)),
                    _ => None,
                })
                .collect::<Vec<_>>();

            if !suggestable_variants_with_placeholders.is_empty() {
                let msg = match (
                    suggestable_variants.is_empty(),
                    suggestable_variants_with_placeholders.len(),
                ) {
                    (true, 1) => "the following enum variant is available",
                    (true, _) => "the following enum variants are available",
                    (false, 1) => "alternatively, the following enum variant is available",
                    (false, _) => "alternatively, the following enum variants are also available",
                };

                err.span_suggestions(
                    span,
                    msg,
                    suggestable_variants_with_placeholders.drain(..),
                    Applicability::HasPlaceholders,
                );
            }
        };

        if def_id.is_local() {
            if let Some(span) = self.def_span(def_id) {
                err.span_note(span, "the enum is defined here");
            }
        }
    }

    crate fn report_missing_type_error(
        &self,
        path: &[Segment],
    ) -> Option<(Span, &'static str, String, Applicability)> {
        let (ident, span) = match path {
            [segment] if !segment.has_generic_args => {
                (segment.ident.to_string(), segment.ident.span)
            }
            _ => return None,
        };
        let mut iter = ident.chars().map(|c| c.is_uppercase());
        let single_uppercase_char =
            matches!(iter.next(), Some(true)) && matches!(iter.next(), None);
        if !self.diagnostic_metadata.currently_processing_generics && !single_uppercase_char {
            return None;
        }
        match (self.diagnostic_metadata.current_item, single_uppercase_char) {
            (Some(Item { kind: ItemKind::Fn(..), ident, .. }), _) if ident.name == sym::main => {
                // Ignore `fn main()` as we don't want to suggest `fn main<T>()`
            }
            (
                Some(Item {
                    kind:
                        kind @ ItemKind::Fn(..)
                        | kind @ ItemKind::Enum(..)
                        | kind @ ItemKind::Struct(..)
                        | kind @ ItemKind::Union(..),
                    ..
                }),
                true,
            )
            | (Some(Item { kind, .. }), false) => {
                // Likely missing type parameter.
                if let Some(generics) = kind.generics() {
                    if span.overlaps(generics.span) {
                        // Avoid the following:
                        // error[E0405]: cannot find trait `A` in this scope
                        //  --> $DIR/typo-suggestion-named-underscore.rs:CC:LL
                        //   |
                        // L | fn foo<T: A>(x: T) {} // Shouldn't suggest underscore
                        //   |           ^- help: you might be missing a type parameter: `, A`
                        //   |           |
                        //   |           not found in this scope
                        return None;
                    }
                    let msg = "you might be missing a type parameter";
                    let (span, sugg) = if let [.., param] = &generics.params[..] {
                        let span = if let [.., bound] = &param.bounds[..] {
                            bound.span()
                        } else {
                            param.ident.span
                        };
                        (span, format!(", {}", ident))
                    } else {
                        (generics.span, format!("<{}>", ident))
                    };
                    // Do not suggest if this is coming from macro expansion.
                    if !span.from_expansion() {
                        return Some((
                            span.shrink_to_hi(),
                            msg,
                            sugg,
                            Applicability::MaybeIncorrect,
                        ));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Given the target `label`, search the `rib_index`th label rib for similarly named labels,
    /// optionally returning the closest match and whether it is reachable.
    crate fn suggestion_for_label_in_rib(
        &self,
        rib_index: usize,
        label: Ident,
    ) -> Option<LabelSuggestion> {
        // Are ribs from this `rib_index` within scope?
        let within_scope = self.is_label_valid_from_rib(rib_index);

        let rib = &self.label_ribs[rib_index];
        let names = rib
            .bindings
            .iter()
            .filter(|(id, _)| id.span.ctxt() == label.span.ctxt())
            .map(|(id, _)| id.name)
            .collect::<Vec<Symbol>>();

        find_best_match_for_name(&names, label.name, None).map(|symbol| {
            // Upon finding a similar name, get the ident that it was from - the span
            // contained within helps make a useful diagnostic. In addition, determine
            // whether this candidate is within scope.
            let (ident, _) = rib.bindings.iter().find(|(ident, _)| ident.name == symbol).unwrap();
            (*ident, within_scope)
        })
    }
}

impl<'tcx> LifetimeContext<'_, 'tcx> {
    crate fn report_missing_lifetime_specifiers(
        &self,
        span: Span,
        count: usize,
    ) -> DiagnosticBuilder<'tcx> {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0106,
            "missing lifetime specifier{}",
            pluralize!(count)
        )
    }

    crate fn emit_undeclared_lifetime_error(&self, lifetime_ref: &hir::Lifetime) {
        let mut err = struct_span_err!(
            self.tcx.sess,
            lifetime_ref.span,
            E0261,
            "use of undeclared lifetime name `{}`",
            lifetime_ref
        );
        err.span_label(lifetime_ref.span, "undeclared lifetime");
        let mut suggests_in_band = false;
        for missing in &self.missing_named_lifetime_spots {
            match missing {
                MissingLifetimeSpot::Generics(generics) => {
                    let (span, sugg) = if let Some(param) = generics.params.iter().find(|p| {
                        !matches!(p.kind, hir::GenericParamKind::Type {
                            synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
                            ..
                        } | hir::GenericParamKind::Lifetime {
                            kind: hir::LifetimeParamKind::Elided,
                        })
                    }) {
                        (param.span.shrink_to_lo(), format!("{}, ", lifetime_ref))
                    } else {
                        suggests_in_band = true;
                        (generics.span, format!("<{}>", lifetime_ref))
                    };
                    err.span_suggestion(
                        span,
                        &format!("consider introducing lifetime `{}` here", lifetime_ref),
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                }
                MissingLifetimeSpot::HigherRanked { span, span_type } => {
                    err.span_suggestion(
                        *span,
                        &format!(
                            "consider making the {} lifetime-generic with a new `{}` lifetime",
                            span_type.descr(),
                            lifetime_ref
                        ),
                        span_type.suggestion(&lifetime_ref.to_string()),
                        Applicability::MaybeIncorrect,
                    );
                    err.note(
                        "for more information on higher-ranked polymorphism, visit \
                            https://doc.rust-lang.org/nomicon/hrtb.html",
                    );
                }
                _ => {}
            }
        }
        if self.tcx.sess.is_nightly_build()
            && !self.tcx.features().in_band_lifetimes
            && suggests_in_band
        {
            err.help(
                "if you want to experiment with in-band lifetime bindings, \
                    add `#![feature(in_band_lifetimes)]` to the crate attributes",
            );
        }
        err.emit();
    }

    // FIXME(const_generics): This patches over a ICE caused by non-'static lifetimes in const
    // generics. We are disallowing this until we can decide on how we want to handle non-'static
    // lifetimes in const generics. See issue #74052 for discussion.
    crate fn emit_non_static_lt_in_const_generic_error(&self, lifetime_ref: &hir::Lifetime) {
        let mut err = struct_span_err!(
            self.tcx.sess,
            lifetime_ref.span,
            E0771,
            "use of non-static lifetime `{}` in const generic",
            lifetime_ref
        );
        err.note(
            "for more information, see issue #74052 \
            <https://github.com/rust-lang/rust/issues/74052>",
        );
        err.emit();
    }

    crate fn is_trait_ref_fn_scope(&mut self, trait_ref: &'tcx hir::PolyTraitRef<'tcx>) -> bool {
        if let def::Res::Def(_, did) = trait_ref.trait_ref.path.res {
            if [
                self.tcx.lang_items().fn_once_trait(),
                self.tcx.lang_items().fn_trait(),
                self.tcx.lang_items().fn_mut_trait(),
            ]
            .contains(&Some(did))
            {
                let (span, span_type) = match &trait_ref.bound_generic_params {
                    [] => (trait_ref.span.shrink_to_lo(), ForLifetimeSpanType::BoundEmpty),
                    [.., bound] => (bound.span.shrink_to_hi(), ForLifetimeSpanType::BoundTail),
                };
                self.missing_named_lifetime_spots
                    .push(MissingLifetimeSpot::HigherRanked { span, span_type });
                return true;
            }
        };
        false
    }

    crate fn add_missing_lifetime_specifiers_label(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        count: usize,
        lifetime_names: &FxHashSet<Symbol>,
        lifetime_spans: Vec<Span>,
        params: &[ElisionFailureInfo],
    ) {
        let snippet = self.tcx.sess.source_map().span_to_snippet(span).ok();

        err.span_label(
            span,
            &format!(
                "expected {} lifetime parameter{}",
                if count == 1 { "named".to_string() } else { count.to_string() },
                pluralize!(count)
            ),
        );

        let suggest_existing = |err: &mut DiagnosticBuilder<'_>,
                                name: &str,
                                formatter: &dyn Fn(&str) -> String| {
            if let Some(MissingLifetimeSpot::HigherRanked { span: for_span, span_type }) =
                self.missing_named_lifetime_spots.iter().rev().next()
            {
                // When we have `struct S<'a>(&'a dyn Fn(&X) -> &X);` we want to not only suggest
                // using `'a`, but also introduce the concept of HRLTs by suggesting
                // `struct S<'a>(&'a dyn for<'b> Fn(&X) -> &'b X);`. (#72404)
                let mut introduce_suggestion = vec![];

                let a_to_z_repeat_n = |n| {
                    (b'a'..=b'z').map(move |c| {
                        let mut s = '\''.to_string();
                        s.extend(std::iter::repeat(char::from(c)).take(n));
                        s
                    })
                };

                // If all single char lifetime names are present, we wrap around and double the chars.
                let lt_name = (1..)
                    .flat_map(a_to_z_repeat_n)
                    .find(|lt| !lifetime_names.contains(&Symbol::intern(&lt)))
                    .unwrap();
                let msg = format!(
                    "consider making the {} lifetime-generic with a new `{}` lifetime",
                    span_type.descr(),
                    lt_name,
                );
                err.note(
                    "for more information on higher-ranked polymorphism, visit \
                    https://doc.rust-lang.org/nomicon/hrtb.html",
                );
                let for_sugg = span_type.suggestion(&lt_name);
                for param in params {
                    if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(param.span) {
                        if snippet.starts_with('&') && !snippet.starts_with("&'") {
                            introduce_suggestion
                                .push((param.span, format!("&{} {}", lt_name, &snippet[1..])));
                        } else if let Some(stripped) = snippet.strip_prefix("&'_ ") {
                            introduce_suggestion
                                .push((param.span, format!("&{} {}", lt_name, stripped)));
                        }
                    }
                }
                introduce_suggestion.push((*for_span, for_sugg));
                introduce_suggestion.push((span, formatter(&lt_name)));
                err.multipart_suggestion(&msg, introduce_suggestion, Applicability::MaybeIncorrect);
            }

            err.span_suggestion_verbose(
                span,
                &format!("consider using the `{}` lifetime", lifetime_names.iter().next().unwrap()),
                formatter(name),
                Applicability::MaybeIncorrect,
            );
        };
        let suggest_new = |err: &mut DiagnosticBuilder<'_>, sugg: &str| {
            for missing in self.missing_named_lifetime_spots.iter().rev() {
                let mut introduce_suggestion = vec![];
                let msg;
                let should_break;
                introduce_suggestion.push(match missing {
                    MissingLifetimeSpot::Generics(generics) => {
                        if generics.span == DUMMY_SP {
                            // Account for malformed generics in the HIR. This shouldn't happen,
                            // but if we make a mistake elsewhere, mainly by keeping something in
                            // `missing_named_lifetime_spots` that we shouldn't, like associated
                            // `const`s or making a mistake in the AST lowering we would provide
                            // non-sensical suggestions. Guard against that by skipping these.
                            // (#74264)
                            continue;
                        }
                        msg = "consider introducing a named lifetime parameter".to_string();
                        should_break = true;
                        if let Some(param) = generics.params.iter().find(|p| {
                            !matches!(p.kind, hir::GenericParamKind::Type {
                                synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
                                ..
                            })
                        }) {
                            (param.span.shrink_to_lo(), "'a, ".to_string())
                        } else {
                            (generics.span, "<'a>".to_string())
                        }
                    }
                    MissingLifetimeSpot::HigherRanked { span, span_type } => {
                        msg = format!(
                            "consider making the {} lifetime-generic with a new `'a` lifetime",
                            span_type.descr(),
                        );
                        should_break = false;
                        err.note(
                            "for more information on higher-ranked polymorphism, visit \
                            https://doc.rust-lang.org/nomicon/hrtb.html",
                        );
                        (*span, span_type.suggestion("'a"))
                    }
                    MissingLifetimeSpot::Static => {
                        let (span, sugg) = match snippet.as_deref() {
                            Some("&") => (span.shrink_to_hi(), "'static ".to_owned()),
                            Some("'_") => (span, "'static".to_owned()),
                            Some(snippet) if !snippet.ends_with('>') => {
                                if snippet == "" {
                                    (
                                        span,
                                        std::iter::repeat("'static")
                                            .take(count)
                                            .collect::<Vec<_>>()
                                            .join(", "),
                                    )
                                } else {
                                    (
                                        span.shrink_to_hi(),
                                        format!(
                                            "<{}>",
                                            std::iter::repeat("'static")
                                                .take(count)
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        ),
                                    )
                                }
                            }
                            _ => continue,
                        };
                        err.span_suggestion_verbose(
                            span,
                            "consider using the `'static` lifetime",
                            sugg.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                        continue;
                    }
                });
                for param in params {
                    if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(param.span) {
                        if snippet.starts_with('&') && !snippet.starts_with("&'") {
                            introduce_suggestion
                                .push((param.span, format!("&'a {}", &snippet[1..])));
                        } else if let Some(stripped) = snippet.strip_prefix("&'_ ") {
                            introduce_suggestion.push((param.span, format!("&'a {}", &stripped)));
                        }
                    }
                }
                introduce_suggestion.push((span, sugg.to_string()));
                err.multipart_suggestion(&msg, introduce_suggestion, Applicability::MaybeIncorrect);
                if should_break {
                    break;
                }
            }
        };

        let lifetime_names: Vec<_> = lifetime_names.iter().collect();
        match (&lifetime_names[..], snippet.as_deref()) {
            ([name], Some("&")) => {
                suggest_existing(err, &name.as_str()[..], &|name| format!("&{} ", name));
            }
            ([name], Some("'_")) => {
                suggest_existing(err, &name.as_str()[..], &|n| n.to_string());
            }
            ([name], Some("")) => {
                suggest_existing(err, &name.as_str()[..], &|n| format!("{}, ", n).repeat(count));
            }
            ([name], Some(snippet)) if !snippet.ends_with('>') => {
                let f = |name: &str| {
                    format!(
                        "{}<{}>",
                        snippet,
                        std::iter::repeat(name.to_string())
                            .take(count)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                suggest_existing(err, &name.as_str()[..], &f);
            }
            ([], Some("&")) if count == 1 => {
                suggest_new(err, "&'a ");
            }
            ([], Some("'_")) if count == 1 => {
                suggest_new(err, "'a");
            }
            ([], Some(snippet)) if !snippet.ends_with('>') => {
                if snippet == "" {
                    // This happens when we have `type Bar<'a> = Foo<T>` where we point at the space
                    // before `T`. We will suggest `type Bar<'a> = Foo<'a, T>`.
                    suggest_new(
                        err,
                        &std::iter::repeat("'a, ").take(count).collect::<Vec<_>>().join(""),
                    );
                } else {
                    suggest_new(
                        err,
                        &format!(
                            "{}<{}>",
                            snippet,
                            std::iter::repeat("'a").take(count).collect::<Vec<_>>().join(", ")
                        ),
                    );
                }
            }
            (lts, ..) if lts.len() > 1 => {
                err.span_note(lifetime_spans, "these named lifetimes are available to use");
                if Some("") == snippet.as_deref() {
                    // This happens when we have `Foo<T>` where we point at the space before `T`,
                    // but this can be confusing so we give a suggestion with placeholders.
                    err.span_suggestion_verbose(
                        span,
                        "consider using one of the available lifetimes here",
                        "'lifetime, ".repeat(count),
                        Applicability::HasPlaceholders,
                    );
                }
            }
            _ => {}
        }
    }

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `const_generics` is not enabled, the body identified by
    /// `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    crate fn maybe_emit_forbidden_non_static_lifetime_error(
        &self,
        body_id: hir::BodyId,
        lifetime_ref: &'tcx hir::Lifetime,
    ) {
        let is_anon_const = matches!(
            self.tcx.def_kind(self.tcx.hir().body_owner_def_id(body_id)),
            hir::def::DefKind::AnonConst
        );
        let is_allowed_lifetime = matches!(
            lifetime_ref.name,
            hir::LifetimeName::Implicit | hir::LifetimeName::Static | hir::LifetimeName::Underscore
        );

        if !self.tcx.lazy_normalization() && is_anon_const && !is_allowed_lifetime {
            feature_err(
                &self.tcx.sess.parse_sess,
                sym::const_generics,
                lifetime_ref.span,
                "a non-static lifetime is not allowed in a `const`",
            )
            .emit();
        }
    }
}
