use crate::diagnostics::{ImportSuggestion, LabelSuggestion, TypoSuggestion};
use crate::late::{AliasPossibility, LateResolutionVisitor, RibKind};
use crate::late::{LifetimeBinderKind, LifetimeRes, LifetimeRibKind, LifetimeUseSet};
use crate::path_names_to_string;
use crate::{Module, ModuleKind, ModuleOrUniformRoot};
use crate::{PathResult, PathSource, Segment};

use rustc_ast::visit::{FnCtxt, FnKind, LifetimeCtxt};
use rustc_ast::{
    self as ast, AssocItemKind, Expr, ExprKind, GenericParam, GenericParamKind, Item, ItemKind,
    MethodCall, NodeId, Path, Ty, TyKind, DUMMY_NODE_ID,
};
use rustc_ast_pretty::pprust::path_segment_to_string;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{
    pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::{DefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::PrimTy;
use rustc_middle::ty::DefIdTree;
use rustc_session::lint;
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::MacroKind;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, Span};

use std::iter;
use std::ops::Deref;

use thin_vec::ThinVec;

type Res = def::Res<ast::NodeId>;

/// A field or associated item from self type suggested in case of resolution failure.
enum AssocSuggestion {
    Field,
    MethodWithSelf { called: bool },
    AssocFn { called: bool },
    AssocType,
    AssocConst,
}

impl AssocSuggestion {
    fn action(&self) -> &'static str {
        match self {
            AssocSuggestion::Field => "use the available field",
            AssocSuggestion::MethodWithSelf { called: true } => {
                "call the method with the fully-qualified path"
            }
            AssocSuggestion::MethodWithSelf { called: false } => {
                "refer to the method with the fully-qualified path"
            }
            AssocSuggestion::AssocFn { called: true } => "call the associated function",
            AssocSuggestion::AssocFn { called: false } => "refer to the associated function",
            AssocSuggestion::AssocConst => "use the associated `const`",
            AssocSuggestion::AssocType => "use the associated type",
        }
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
        segments: suggestion.path.segments[0..path_len - 1].iter().cloned().collect(),
        tokens: None,
    };
    let enum_path_string = path_names_to_string(&enum_path);

    (variant_path_string, enum_path_string)
}

/// Description of an elided lifetime.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) struct MissingLifetime {
    /// Used to overwrite the resolution with the suggestion, to avoid cascasing errors.
    pub id: NodeId,
    /// Where to suggest adding the lifetime.
    pub span: Span,
    /// How the lifetime was introduced, to have the correct space and comma.
    pub kind: MissingLifetimeKind,
    /// Number of elided lifetimes, used for elision in path.
    pub count: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum MissingLifetimeKind {
    /// An explicit `'_`.
    Underscore,
    /// An elided lifetime `&' ty`.
    Ampersand,
    /// An elided lifetime in brackets with written brackets.
    Comma,
    /// An elided lifetime with elided brackets.
    Brackets,
}

/// Description of the lifetimes appearing in a function parameter.
/// This is used to provide a literal explanation to the elision failure.
#[derive(Clone, Debug)]
pub(super) struct ElisionFnParameter {
    /// The index of the argument in the original definition.
    pub index: usize,
    /// The name of the argument if it's a simple ident.
    pub ident: Option<Ident>,
    /// The number of lifetimes in the parameter.
    pub lifetime_count: usize,
    /// The span of the parameter.
    pub span: Span,
}

/// Description of lifetimes that appear as candidates for elision.
/// This is used to suggest introducing an explicit lifetime.
#[derive(Debug)]
pub(super) enum LifetimeElisionCandidate {
    /// This is not a real lifetime.
    Ignore,
    /// There is a named lifetime, we won't suggest anything.
    Named,
    Missing(MissingLifetime),
}

/// Only used for diagnostics.
#[derive(Debug)]
struct BaseError {
    msg: String,
    fallback_label: String,
    span: Span,
    span_label: Option<(Span, &'static str)>,
    could_be_expr: bool,
    suggestion: Option<(Span, &'static str, String)>,
}

#[derive(Debug)]
enum TypoCandidate {
    Typo(TypoSuggestion),
    Shadowed(Res, Option<Span>),
    None,
}

impl TypoCandidate {
    fn to_opt_suggestion(self) -> Option<TypoSuggestion> {
        match self {
            TypoCandidate::Typo(sugg) => Some(sugg),
            TypoCandidate::Shadowed(_, _) | TypoCandidate::None => None,
        }
    }
}

impl<'a: 'ast, 'ast, 'tcx> LateResolutionVisitor<'a, '_, 'ast, 'tcx> {
    fn def_span(&self, def_id: DefId) -> Option<Span> {
        match def_id.krate {
            LOCAL_CRATE => self.r.opt_span(def_id),
            _ => Some(self.r.cstore().get_span_untracked(def_id, self.r.session)),
        }
    }

    fn make_base_error(
        &mut self,
        path: &[Segment],
        span: Span,
        source: PathSource<'_>,
        res: Option<Res>,
    ) -> BaseError {
        // Make the base error.
        let mut expected = source.descr_expected();
        let path_str = Segment::names_to_string(path);
        let item_str = path.last().unwrap().ident;
        if let Some(res) = res {
            BaseError {
                msg: format!("expected {}, found {} `{}`", expected, res.descr(), path_str),
                fallback_label: format!("not a {expected}"),
                span,
                span_label: match res {
                    Res::Def(kind, def_id) if kind == DefKind::TyParam => {
                        self.def_span(def_id).map(|span| (span, "found this type parameter"))
                    }
                    _ => None,
                },
                could_be_expr: match res {
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
                suggestion: None,
            }
        } else {
            let item_span = path.last().unwrap().ident.span;
            let (mod_prefix, mod_str, suggestion) = if path.len() == 1 {
                debug!(?self.diagnostic_metadata.current_impl_items);
                debug!(?self.diagnostic_metadata.current_function);
                let suggestion = if self.current_trait_ref.is_none()
                    && let Some((fn_kind, _)) = self.diagnostic_metadata.current_function
                    && let Some(FnCtxt::Assoc(_)) = fn_kind.ctxt()
                    && let Some(items) = self.diagnostic_metadata.current_impl_items
                    && let Some(item) = items.iter().find(|i| {
                        if let AssocItemKind::Fn(..) | AssocItemKind::Const(..) = &i.kind
                            && i.ident.name == item_str.name
                        {
                            debug!(?item_str.name);
                            return true
                        }
                        false
                    })
                {
                    let self_sugg = match &item.kind {
                        AssocItemKind::Fn(fn_) if fn_.sig.decl.has_self() => "self.",
                        _ => "Self::",
                    };

                    Some((
                        item_span.shrink_to_lo(),
                        match &item.kind {
                            AssocItemKind::Fn(..) => "consider using the associated function",
                            AssocItemKind::Const(..) => "consider using the associated constant",
                            _ => unreachable!("item kind was filtered above"),
                        },
                        self_sugg.to_string()
                    ))
                } else {
                    None
                };
                (String::new(), "this scope".to_string(), suggestion)
            } else if path.len() == 2 && path[0].ident.name == kw::PathRoot {
                if self.r.session.edition() > Edition::Edition2015 {
                    // In edition 2018 onwards, the `::foo` syntax may only pull from the extern prelude
                    // which overrides all other expectations of item type
                    expected = "crate";
                    (String::new(), "the list of imported crates".to_string(), None)
                } else {
                    (String::new(), "the crate root".to_string(), None)
                }
            } else if path.len() == 2 && path[0].ident.name == kw::Crate {
                (String::new(), "the crate root".to_string(), None)
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_prefix = match self.resolve_path(mod_path, Some(TypeNS), None) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => module.res(),
                    _ => None,
                }
                .map_or_else(String::new, |res| format!("{} ", res.descr()));
                (mod_prefix, format!("`{}`", Segment::names_to_string(mod_path)), None)
            };

            let (fallback_label, suggestion) = if path_str == "async"
                && expected.starts_with("struct")
            {
                ("`async` blocks are only allowed in Rust 2018 or later".to_string(), suggestion)
            } else {
                // check if we are in situation of typo like `True` instead of `true`.
                let override_suggestion =
                    if ["true", "false"].contains(&item_str.to_string().to_lowercase().as_str()) {
                        let item_typo = item_str.to_string().to_lowercase();
                        Some((item_span, "you may want to use a bool value instead", item_typo))
                    // FIXME(vincenzopalazzo): make the check smarter,
                    // and maybe expand with levenshtein distance checks
                    } else if item_str.as_str() == "printf" {
                        Some((
                            item_span,
                            "you may have meant to use the `print` macro",
                            "print!".to_owned(),
                        ))
                    } else {
                        suggestion
                    };
                (format!("not found in {mod_str}"), override_suggestion)
            };

            BaseError {
                msg: format!("cannot find {expected} `{item_str}` in {mod_prefix}{mod_str}"),
                fallback_label,
                span: item_span,
                span_label: None,
                could_be_expr: false,
                suggestion,
            }
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
    ) -> (DiagnosticBuilder<'tcx, ErrorGuaranteed>, Vec<ImportSuggestion>) {
        debug!(?res, ?source);
        let base_error = self.make_base_error(path, span, source, res);
        let code = source.error_code(res.is_some());
        let mut err =
            self.r.session.struct_span_err_with_code(base_error.span, &base_error.msg, code);

        self.suggest_swapping_misplaced_self_ty_and_trait(&mut err, source, res, base_error.span);

        if let Some((span, label)) = base_error.span_label {
            err.span_label(span, label);
        }

        if let Some(ref sugg) = base_error.suggestion {
            err.span_suggestion_verbose(sugg.0, sugg.1, &sugg.2, Applicability::MaybeIncorrect);
        }

        self.suggest_bare_struct_literal(&mut err);

        if self.suggest_pattern_match_with_let(&mut err, source, span) {
            // Fallback label.
            err.span_label(base_error.span, &base_error.fallback_label);
            return (err, Vec::new());
        }

        self.suggest_self_or_self_ref(&mut err, path, span);
        self.detect_assoct_type_constraint_meant_as_path(&mut err, &base_error);
        if self.suggest_self_ty(&mut err, source, path, span)
            || self.suggest_self_value(&mut err, source, path, span)
        {
            return (err, Vec::new());
        }

        let (found, candidates) =
            self.try_lookup_name_relaxed(&mut err, source, path, span, res, &base_error);
        if found {
            return (err, candidates);
        }

        if !self.type_ascription_suggestion(&mut err, base_error.span) {
            let mut fallback =
                self.suggest_trait_and_bounds(&mut err, source, res, span, &base_error);

            // if we have suggested using pattern matching, then don't add needless suggestions
            // for typos.
            fallback |= self.suggest_typo(&mut err, source, path, span, &base_error);

            if fallback {
                // Fallback label.
                err.span_label(base_error.span, &base_error.fallback_label);
            }
        }
        self.err_code_special_cases(&mut err, source, path, span);

        (err, candidates)
    }

    fn detect_assoct_type_constraint_meant_as_path(
        &self,
        err: &mut Diagnostic,
        base_error: &BaseError,
    ) {
        let Some(ty) = self.diagnostic_metadata.current_type_path else { return; };
        let TyKind::Path(_, path) = &ty.kind else { return; };
        for segment in &path.segments {
            let Some(params) = &segment.args else { continue; };
            let ast::GenericArgs::AngleBracketed(ref params) = params.deref() else { continue; };
            for param in &params.args {
                let ast::AngleBracketedArg::Constraint(constraint) = param else { continue; };
                let ast::AssocConstraintKind::Bound { bounds } = &constraint.kind else {
                    continue;
                };
                for bound in bounds {
                    let ast::GenericBound::Trait(trait_ref, ast::TraitBoundModifier::None)
                        = bound else
                    {
                        continue;
                    };
                    if base_error.span == trait_ref.span {
                        err.span_suggestion_verbose(
                            constraint.ident.span.between(trait_ref.span),
                            "you might have meant to write a path instead of an associated type bound",
                            "::",
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }

    fn suggest_self_or_self_ref(&mut self, err: &mut Diagnostic, path: &[Segment], span: Span) {
        if !self.self_type_is_available() {
            return;
        }
        let Some(path_last_segment) = path.last() else { return };
        let item_str = path_last_segment.ident;
        // Emit help message for fake-self from other languages (e.g., `this` in Javascript).
        if ["this", "my"].contains(&item_str.as_str()) {
            err.span_suggestion_short(
                span,
                "you might have meant to use `self` here instead",
                "self",
                Applicability::MaybeIncorrect,
            );
            if !self.self_value_is_available(path[0].ident.span) {
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
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    fn try_lookup_name_relaxed(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
        res: Option<Res>,
        base_error: &BaseError,
    ) -> (bool, Vec<ImportSuggestion>) {
        // Try to lookup name in more relaxed fashion for better error reporting.
        let ident = path.last().unwrap().ident;
        let is_expected = &|res| source.is_expected(res);
        let ns = source.namespace();
        let is_enum_variant = &|res| matches!(res, Res::Def(DefKind::Variant, _));
        let path_str = Segment::names_to_string(path);
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let mut candidates = self
            .r
            .lookup_import_candidates(ident, ns, &self.parent_scope, is_expected)
            .into_iter()
            .filter(|ImportSuggestion { did, .. }| {
                match (did, res.and_then(|res| res.opt_def_id())) {
                    (Some(suggestion_did), Some(actual_did)) => *suggestion_did != actual_did,
                    _ => true,
                }
            })
            .collect::<Vec<_>>();
        let crate_def_id = CRATE_DEF_ID.to_def_id();
        // Try to filter out intrinsics candidates, as long as we have
        // some other candidates to suggest.
        let intrinsic_candidates: Vec<_> = candidates
            .drain_filter(|sugg| {
                let path = path_names_to_string(&sugg.path);
                path.starts_with("core::intrinsics::") || path.starts_with("std::intrinsics::")
            })
            .collect();
        if candidates.is_empty() {
            // Put them back if we have no more candidates to suggest...
            candidates.extend(intrinsic_candidates);
        }
        if candidates.is_empty() && is_expected(Res::Def(DefKind::Enum, crate_def_id)) {
            let mut enum_candidates: Vec<_> = self
                .r
                .lookup_import_candidates(ident, ns, &self.parent_scope, is_enum_variant)
                .into_iter()
                .map(|suggestion| import_candidate_to_enum_paths(&suggestion))
                .filter(|(_, enum_ty_path)| !enum_ty_path.starts_with("std::prelude::"))
                .collect();
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
                        err.downgrade_to_delayed_bug();
                        return (true, candidates);
                    }
                }

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
                    enum_candidates.into_iter().map(|(_variant_path, enum_ty_path)| enum_ty_path),
                    Applicability::MachineApplicable,
                );
            }
        }

        // Try Levenshtein algorithm.
        let typo_sugg =
            self.lookup_typo_candidate(path, source.namespace(), is_expected).to_opt_suggestion();
        if path.len() == 1 && self.self_type_is_available() {
            if let Some(candidate) =
                self.lookup_assoc_candidate(ident, ns, is_expected, source.is_call())
            {
                let self_is_available = self.self_value_is_available(path[0].ident.span);
                match candidate {
                    AssocSuggestion::Field => {
                        if self_is_available {
                            err.span_suggestion(
                                span,
                                "you might have meant to use the available field",
                                format!("self.{path_str}"),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_label(span, "a field by this name exists in `Self`");
                        }
                    }
                    AssocSuggestion::MethodWithSelf { called } if self_is_available => {
                        let msg = if called {
                            "you might have meant to call the method"
                        } else {
                            "you might have meant to refer to the method"
                        };
                        err.span_suggestion(
                            span,
                            msg,
                            format!("self.{path_str}"),
                            Applicability::MachineApplicable,
                        );
                    }
                    AssocSuggestion::MethodWithSelf { .. }
                    | AssocSuggestion::AssocFn { .. }
                    | AssocSuggestion::AssocConst
                    | AssocSuggestion::AssocType => {
                        err.span_suggestion(
                            span,
                            &format!("you might have meant to {}", candidate.action()),
                            format!("Self::{path_str}"),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                self.r.add_typo_suggestion(err, typo_sugg, ident_span);
                return (true, candidates);
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
                    &format!("try calling `{ident}` as a method"),
                    format!("self.{path_str}({args_snippet})"),
                    Applicability::MachineApplicable,
                );
                return (true, candidates);
            }
        }

        // Try context-dependent help if relaxed lookup didn't work.
        if let Some(res) = res {
            if self.smart_resolve_context_dependent_help(
                err,
                span,
                source,
                res,
                &path_str,
                &base_error.fallback_label,
            ) {
                // We do this to avoid losing a secondary span when we override the main error span.
                self.r.add_typo_suggestion(err, typo_sugg, ident_span);
                return (true, candidates);
            }
        }

        // Try to find in last block rib
        if let Some(rib) = &self.last_block_rib && let RibKind::NormalRibKind = rib.kind {
            for (ident, &res) in &rib.bindings {
                if let Res::Local(_) = res && path.len() == 1 &&
                    ident.span.eq_ctxt(path[0].ident.span) &&
                    ident.name == path[0].ident.name {
                    err.span_help(
                        ident.span,
                        &format!("the binding `{}` is available in a different scope in the same function", path_str),
                    );
                    return (true, candidates);
                }
            }
        }

        return (false, candidates);
    }

    fn suggest_trait_and_bounds(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        res: Option<Res>,
        span: Span,
        base_error: &BaseError,
    ) -> bool {
        let is_macro =
            base_error.span.from_expansion() && base_error.span.desugaring_kind().is_none();
        let mut fallback = false;

        if let (
            PathSource::Trait(AliasPossibility::Maybe),
            Some(Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _)),
            false,
        ) = (source, res, is_macro)
        {
            if let Some(bounds @ [_, .., _]) = self.diagnostic_metadata.current_trait_object {
                fallback = true;
                let spans: Vec<Span> = bounds
                    .iter()
                    .map(|bound| bound.span())
                    .filter(|&sp| sp != base_error.span)
                    .collect();

                let start_span = bounds[0].span();
                // `end_span` is the end of the poly trait ref (Foo + 'baz + Bar><)
                let end_span = bounds.last().unwrap().span();
                // `last_bound_span` is the last bound of the poly trait ref (Foo + >'baz< + Bar)
                let last_bound_span = spans.last().cloned().unwrap();
                let mut multi_span: MultiSpan = spans.clone().into();
                for sp in spans {
                    let msg = if sp == last_bound_span {
                        format!(
                            "...because of {these} bound{s}",
                            these = pluralize!("this", bounds.len() - 1),
                            s = pluralize!(bounds.len() - 1),
                        )
                    } else {
                        String::new()
                    };
                    multi_span.push_span_label(sp, msg);
                }
                multi_span.push_span_label(base_error.span, "expected this type to be a trait...");
                err.span_help(
                    multi_span,
                    "`+` is used to constrain a \"trait object\" type with lifetimes or \
                        auto-traits; structs and enums can't be bound in that way",
                );
                if bounds.iter().all(|bound| match bound {
                    ast::GenericBound::Outlives(_) => true,
                    ast::GenericBound::Trait(tr, _) => tr.span == base_error.span,
                }) {
                    let mut sugg = vec![];
                    if base_error.span != start_span {
                        sugg.push((start_span.until(base_error.span), String::new()));
                    }
                    if base_error.span != end_span {
                        sugg.push((base_error.span.shrink_to_hi().to(end_span), String::new()));
                    }

                    err.multipart_suggestion(
                        "if you meant to use a type and not a trait here, remove the bounds",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }

        fallback |= self.restrict_assoc_type_in_where_clause(span, err);
        fallback
    }

    fn suggest_typo(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
        base_error: &BaseError,
    ) -> bool {
        let is_expected = &|res| source.is_expected(res);
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let typo_sugg = self.lookup_typo_candidate(path, source.namespace(), is_expected);
        let is_in_same_file = &|sp1, sp2| {
            let source_map = self.r.session.source_map();
            let file1 = source_map.span_to_filename(sp1);
            let file2 = source_map.span_to_filename(sp2);
            file1 == file2
        };
        // print 'you might have meant' if the candidate is (1) is a shadowed name with
        // accessible definition and (2) either defined in the same crate as the typo
        // (could be in a different file) or introduced in the same file as the typo
        // (could belong to a different crate)
        if let TypoCandidate::Shadowed(res, Some(sugg_span)) = typo_sugg
            && res
                .opt_def_id()
                .map_or(false, |id| id.is_local() || is_in_same_file(span, sugg_span))
        {
            err.span_label(
                sugg_span,
                format!("you might have meant to refer to this {}", res.descr()),
            );
            return true;
        }
        let mut fallback = false;
        let typo_sugg = typo_sugg.to_opt_suggestion();
        if !self.r.add_typo_suggestion(err, typo_sugg, ident_span) {
            fallback = true;
            match self.diagnostic_metadata.current_let_binding {
                Some((pat_sp, Some(ty_sp), None))
                    if ty_sp.contains(base_error.span) && base_error.could_be_expr =>
                {
                    err.span_suggestion_short(
                        pat_sp.between(ty_sp),
                        "use `=` if you meant to assign",
                        " = ",
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {}
            }

            // If the trait has a single item (which wasn't matched by Levenshtein), suggest it
            let suggestion = self.get_single_associated_item(&path, &source, is_expected);
            if !self.r.add_typo_suggestion(err, suggestion, ident_span) {
                fallback = !self.let_binding_suggestion(err, ident_span);
            }
        }
        fallback
    }

    fn err_code_special_cases(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) {
        if let Some(err_code) = &err.code {
            if err_code == &rustc_errors::error_code!(E0425) {
                for label_rib in &self.label_ribs {
                    for (label_ident, node_id) in &label_rib.bindings {
                        let ident = path.last().unwrap().ident;
                        if format!("'{}", ident) == label_ident.to_string() {
                            err.span_label(label_ident.span, "a label with a similar name exists");
                            if let PathSource::Expr(Some(Expr {
                                kind: ExprKind::Break(None, Some(_)),
                                ..
                            })) = source
                            {
                                err.span_suggestion(
                                    span,
                                    "use the similarly named label",
                                    label_ident.name,
                                    Applicability::MaybeIncorrect,
                                );
                                // Do not lint against unused label when we suggest them.
                                self.diagnostic_metadata.unused_labels.remove(node_id);
                            }
                        }
                    }
                }
            } else if err_code == &rustc_errors::error_code!(E0412) {
                if let Some(correct) = Self::likely_rust_type(path) {
                    err.span_suggestion(
                        span,
                        "perhaps you intended to use this type",
                        correct,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    /// Emit special messages for unresolved `Self` and `self`.
    fn suggest_self_ty(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) -> bool {
        if !is_self_type(path, source.namespace()) {
            return false;
        }
        err.code(rustc_errors::error_code!(E0411));
        err.span_label(span, "`Self` is only available in impls, traits, and type definitions");
        if let Some(item_kind) = self.diagnostic_metadata.current_item {
            if !item_kind.ident.span.is_dummy() {
                err.span_label(
                    item_kind.ident.span,
                    format!(
                        "`Self` not allowed in {} {}",
                        item_kind.kind.article(),
                        item_kind.kind.descr()
                    ),
                );
            }
        }
        true
    }

    fn suggest_self_value(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) -> bool {
        if !is_self_value(path, source.namespace()) {
            return false;
        }

        debug!("smart_resolve_path_fragment: E0424, source={:?}", source);
        err.code(rustc_errors::error_code!(E0424));
        err.span_label(
            span,
            match source {
                PathSource::Pat => {
                    "`self` value is a keyword and may not be bound to variables or shadowed"
                }
                _ => "`self` value is a keyword only available in methods with a `self` parameter",
            },
        );
        let is_assoc_fn = self.self_type_is_available();
        if let Some((fn_kind, span)) = &self.diagnostic_metadata.current_function {
            // The current function has a `self' parameter, but we were unable to resolve
            // a reference to `self`. This can only happen if the `self` identifier we
            // are resolving came from a different hygiene context.
            if fn_kind.decl().inputs.get(0).map_or(false, |p| p.is_self()) {
                err.span_label(*span, "this function has a `self` parameter, but a macro invocation can only access identifiers it receives from parameters");
            } else {
                let doesnt = if is_assoc_fn {
                    let (span, sugg) = fn_kind
                        .decl()
                        .inputs
                        .get(0)
                        .map(|p| (p.span.shrink_to_lo(), "&self, "))
                        .unwrap_or_else(|| {
                            // Try to look for the "(" after the function name, if possible.
                            // This avoids placing the suggestion into the visibility specifier.
                            let span = fn_kind
                                .ident()
                                .map_or(*span, |ident| span.with_lo(ident.span.hi()));
                            (
                                self.r
                                    .session
                                    .source_map()
                                    .span_through_char(span, '(')
                                    .shrink_to_hi(),
                                "&self",
                            )
                        });
                    err.span_suggestion_verbose(
                        span,
                        "add a `self` receiver parameter to make the associated `fn` a method",
                        sugg,
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
        } else if let Some(item_kind) = self.diagnostic_metadata.current_item {
            err.span_label(
                item_kind.ident.span,
                format!(
                    "`self` not allowed in {} {}",
                    item_kind.kind.article(),
                    item_kind.kind.descr()
                ),
            );
        }
        true
    }

    fn suggest_swapping_misplaced_self_ty_and_trait(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        res: Option<Res>,
        span: Span,
    ) {
        if let Some((trait_ref, self_ty)) =
            self.diagnostic_metadata.currently_processing_impl_trait.clone()
            && let TyKind::Path(_, self_ty_path) = &self_ty.kind
            && let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(&Segment::from_path(self_ty_path), Some(TypeNS), None)
            && let ModuleKind::Def(DefKind::Trait, ..) = module.kind
            && trait_ref.path.span == span
            && let PathSource::Trait(_) = source
            && let Some(Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _)) = res
            && let Ok(self_ty_str) =
                self.r.session.source_map().span_to_snippet(self_ty.span)
            && let Ok(trait_ref_str) =
                self.r.session.source_map().span_to_snippet(trait_ref.path.span)
        {
                err.multipart_suggestion(
                    "`impl` items mention the trait being implemented first and the type it is being implemented for second",
                    vec![(trait_ref.path.span, self_ty_str), (self_ty.span, trait_ref_str)],
                    Applicability::MaybeIncorrect,
                );
        }
    }

    fn suggest_bare_struct_literal(&mut self, err: &mut Diagnostic) {
        if let Some(span) = self.diagnostic_metadata.current_block_could_be_bare_struct_literal {
            err.multipart_suggestion(
                "you might have meant to write a `struct` literal",
                vec![
                    (span.shrink_to_lo(), "{ SomeStruct ".to_string()),
                    (span.shrink_to_hi(), "}".to_string()),
                ],
                Applicability::HasPlaceholders,
            );
        }
    }

    fn suggest_pattern_match_with_let(
        &mut self,
        err: &mut Diagnostic,
        source: PathSource<'_>,
        span: Span,
    ) -> bool {
        if let PathSource::Expr(_) = source &&
        let Some(Expr {
                    span: expr_span,
                    kind: ExprKind::Assign(lhs, _, _),
                    ..
                })  = self.diagnostic_metadata.in_if_condition {
            // Icky heuristic so we don't suggest:
            // `if (i + 2) = 2` => `if let (i + 2) = 2` (approximately pattern)
            // `if 2 = i` => `if let 2 = i` (lhs needs to contain error span)
            if lhs.is_approximately_pattern() && lhs.span.contains(span) {
                err.span_suggestion_verbose(
                    expr_span.shrink_to_lo(),
                    "you might have meant to use pattern matching",
                    "let ",
                    Applicability::MaybeIncorrect,
                );
                return true;
            }
        }
        false
    }

    fn get_single_associated_item(
        &mut self,
        path: &[Segment],
        source: &PathSource<'_>,
        filter_fn: &impl Fn(Res) -> bool,
    ) -> Option<TypoSuggestion> {
        if let crate::PathSource::TraitItem(_) = source {
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(mod_path, None, None)
            {
                let resolutions = self.r.resolutions(module).borrow();
                let targets: Vec<_> =
                    resolutions
                        .iter()
                        .filter_map(|(key, resolution)| {
                            resolution.borrow().binding.map(|binding| binding.res()).and_then(
                                |res| if filter_fn(res) { Some((key, res)) } else { None },
                            )
                        })
                        .collect();
                if targets.len() == 1 {
                    let target = targets[0];
                    return Some(TypoSuggestion::single_item_from_ident(target.0.ident, target.1));
                }
            }
        }
        None
    }

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    fn restrict_assoc_type_in_where_clause(&mut self, span: Span, err: &mut Diagnostic) -> bool {
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
        let (ty, position, path) = if let ast::TyKind::Path(Some(qself), path) = &bounded_ty.kind {
            // use this to verify that ident is a type param.
            let Some(partial_res) = self.r.partial_res_map.get(&bounded_ty.id) else {
                return false;
            };
            if !matches!(
                partial_res.full_res(),
                Some(hir::def::Res::Def(hir::def::DefKind::AssocTy, _))
            ) {
                return false;
            }
            (&qself.ty, qself.position, path)
        } else {
            return false;
        };

        let peeled_ty = ty.peel_refs();
        if let ast::TyKind::Path(None, type_param_path) = &peeled_ty.kind {
            // Confirm that the `SelfTy` is a type parameter.
            let Some(partial_res) = self.r.partial_res_map.get(&peeled_ty.id) else {
                return false;
            };
            if !matches!(
                partial_res.full_res(),
                Some(hir::def::Res::Def(hir::def::DefKind::TyParam, _))
            ) {
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
                                path.segments[..position]
                                    .iter()
                                    .map(|segment| path_segment_to_string(segment))
                                    .collect::<Vec<_>>()
                                    .join("::"),
                                path.segments[position..]
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
                                            None,
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
        let sp = sm.span_look_ahead(span, None, Some(50));
        let followed_by_brace = matches!(sm.span_to_snippet(sp), Ok(ref snippet) if snippet == "{");
        // In case this could be a struct literal that needs to be surrounded
        // by parentheses, find the appropriate span.
        let closing_span = sm.span_look_ahead(span, Some("}"), Some(50));
        let closing_brace: Option<Span> = sm
            .span_to_snippet(closing_span)
            .map_or(None, |s| if s == "}" { Some(span.to(closing_span)) } else { None });
        (followed_by_brace, closing_brace)
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut Diagnostic,
        span: Span,
        source: PathSource<'_>,
        res: Res,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);

        let path_sep = |err: &mut Diagnostic, expr: &Expr, kind: DefKind| {
            const MESSAGE: &str = "use the path separator to refer to an item";

            let (lhs_span, rhs_span) = match &expr.kind {
                ExprKind::Field(base, ident) => (base.span, ident.span),
                ExprKind::MethodCall(box MethodCall { receiver, span, .. }) => {
                    (receiver.span, *span)
                }
                _ => return false,
            };

            if lhs_span.eq_ctxt(rhs_span) {
                err.span_suggestion(
                    lhs_span.between(rhs_span),
                    MESSAGE,
                    "::",
                    Applicability::MaybeIncorrect,
                );
                true
            } else if kind == DefKind::Struct
            && let Some(lhs_source_span) = lhs_span.find_ancestor_inside(expr.span)
            && let Ok(snippet) = self.r.session.source_map().span_to_snippet(lhs_source_span)
            {
                // The LHS is a type that originates from a macro call.
                // We have to add angle brackets around it.

                err.span_suggestion_verbose(
                    lhs_source_span.until(rhs_span),
                    MESSAGE,
                    format!("<{snippet}>::"),
                    Applicability::MaybeIncorrect,
                );
                true
            } else {
                // Either we were unable to obtain the source span / the snippet or
                // the LHS originates from a macro call and it is not a type and thus
                // there is no way to replace `.` with `::` and still somehow suggest
                // valid Rust code.

                false
            }
        };

        let find_span = |source: &PathSource<'_>, err: &mut Diagnostic| {
            match source {
                PathSource::Expr(Some(Expr { span, kind: ExprKind::Call(_, _), .. }))
                | PathSource::TupleStruct(span, _) => {
                    // We want the main underline to cover the suggested code as well for
                    // cleaner output.
                    err.set_span(*span);
                    *span
                }
                _ => span,
            }
        };

        let mut bad_struct_syntax_suggestion = |def_id: DefId| {
            let (followed_by_brace, closing_brace) = self.followed_by_brace(span);

            match source {
                PathSource::Expr(Some(
                    parent @ Expr { kind: ExprKind::Field(..) | ExprKind::MethodCall(..), .. },
                )) if path_sep(err, &parent, DefKind::Struct) => {}
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
                    let span = find_span(&source, err);
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
            (
                Res::Def(DefKind::Macro(MacroKind::Bang), _),
                PathSource::Expr(Some(Expr {
                    kind: ExprKind::Index(..) | ExprKind::Call(..), ..
                }))
                | PathSource::Struct,
            ) => {
                err.span_label(span, fallback_label);
                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "use `!` to invoke the macro",
                    "!",
                    Applicability::MaybeIncorrect,
                );
                if path_str == "try" && span.is_rust_2015() {
                    err.note("if you want the `try` keyword, you need Rust 2018 or later");
                }
            }
            (Res::Def(DefKind::Macro(MacroKind::Bang), _), _) => {
                err.span_label(span, fallback_label);
            }
            (Res::Def(DefKind::TyAlias, def_id), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if self.r.session.is_nightly_build() {
                    let msg = "you might have meant to use `#![feature(trait_alias)]` instead of a \
                               `type` alias";
                    if let Some(span) = self.def_span(def_id) {
                        if let Ok(snip) = self.r.session.source_map().span_to_snippet(span) {
                            // The span contains a type alias so we should be able to
                            // replace `type` with `trait`.
                            let snip = snip.replacen("type", "trait", 1);
                            err.span_suggestion(span, msg, snip, Applicability::MaybeIncorrect);
                        } else {
                            err.span_help(span, msg);
                        }
                    } else {
                        err.help(msg);
                    }
                }
            }
            (
                Res::Def(kind @ (DefKind::Mod | DefKind::Trait), _),
                PathSource::Expr(Some(parent)),
            ) => {
                if !path_sep(err, &parent, kind) {
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
                    err.downgrade_to_delayed_bug();
                    // We already suggested changing `:` into `::` during parsing.
                    return false;
                }

                self.suggest_using_enum_variant(err, source, def_id, span);
            }
            (Res::Def(DefKind::Struct, def_id), source) if ns == ValueNS => {
                let (ctor_def, ctor_vis, fields) =
                    if let Some(struct_ctor) = self.r.struct_constructors.get(&def_id).cloned() {
                        if let PathSource::Expr(Some(parent)) = source {
                            if let ExprKind::Field(..) | ExprKind::MethodCall(..) = parent.kind {
                                bad_struct_syntax_suggestion(def_id);
                                return true;
                            }
                        }
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
                    let non_visible_spans: Vec<Span> = iter::zip(&fields, &spans)
                        .filter(|(vis, _)| {
                            !self.r.is_accessible_from(**vis, self.parent_scope.module)
                        })
                        .map(|(_, span)| *span)
                        .collect();

                    if non_visible_spans.len() > 0 {
                        if let Some(fields) = self.r.field_visibility_spans.get(&def_id) {
                            err.multipart_suggestion_verbose(
                                &format!(
                                    "consider making the field{} publicly accessible",
                                    pluralize!(fields.len())
                                ),
                                fields.iter().map(|span| (*span, "pub ".to_string())).collect(),
                                Applicability::MaybeIncorrect,
                            );
                        }

                        let mut m: MultiSpan = non_visible_spans.clone().into();
                        non_visible_spans
                            .into_iter()
                            .for_each(|s| m.push_span_label(s, "private field"));
                        err.span_note(m, "constructor is not visible here due to private fields");
                    }

                    return true;
                }

                err.span_label(span, "constructor is not visible here due to private fields");
            }
            (Res::Def(DefKind::Union | DefKind::Variant, def_id), _) if ns == ValueNS => {
                bad_struct_syntax_suggestion(def_id);
            }
            (Res::Def(DefKind::Ctor(_, CtorKind::Const), def_id), _) if ns == ValueNS => {
                match source {
                    PathSource::Expr(_) | PathSource::TupleStruct(..) | PathSource::Pat => {
                        let span = find_span(&source, err);
                        if let Some(span) = self.def_span(def_id) {
                            err.span_label(span, &format!("`{}` defined here", path_str));
                        }
                        err.span_suggestion(
                            span,
                            "use this syntax instead",
                            path_str,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    _ => return false,
                }
            }
            (Res::Def(DefKind::Ctor(_, CtorKind::Fn), ctor_def_id), _) if ns == ValueNS => {
                let def_id = self.r.parent(ctor_def_id);
                if let Some(span) = self.def_span(def_id) {
                    err.span_label(span, &format!("`{}` defined here", path_str));
                }
                let fields = self.r.field_names.get(&def_id).map_or_else(
                    || "/* fields */".to_string(),
                    |fields| vec!["_"; fields.len()].join(", "),
                );
                err.span_suggestion(
                    span,
                    "use the tuple variant pattern syntax instead",
                    format!("{}({})", path_str, fields),
                    Applicability::HasPlaceholders,
                );
            }
            (Res::SelfTyParam { .. } | Res::SelfTyAlias { .. }, _) if ns == ValueNS => {
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

    /// Given the target `ident` and `kind`, search for the similarly named associated item
    /// in `self.current_trait_ref`.
    pub(crate) fn find_similarly_named_assoc_item(
        &mut self,
        ident: Symbol,
        kind: &AssocItemKind,
    ) -> Option<Symbol> {
        let (module, _) = self.current_trait_ref.as_ref()?;
        if ident == kw::Underscore {
            // We do nothing for `_`.
            return None;
        }

        let resolutions = self.r.resolutions(module);
        let targets = resolutions
            .borrow()
            .iter()
            .filter_map(|(key, res)| res.borrow().binding.map(|binding| (key, binding.res())))
            .filter(|(_, res)| match (kind, res) {
                (AssocItemKind::Const(..), Res::Def(DefKind::AssocConst, _)) => true,
                (AssocItemKind::Fn(_), Res::Def(DefKind::AssocFn, _)) => true,
                (AssocItemKind::Type(..), Res::Def(DefKind::AssocTy, _)) => true,
                _ => false,
            })
            .map(|(key, _)| key.ident.name)
            .collect::<Vec<_>>();

        find_best_match_for_name(&targets, ident, None)
    }

    fn lookup_assoc_candidate<FilterFn>(
        &mut self,
        ident: Ident,
        ns: Namespace,
        filter_fn: FilterFn,
        called: bool,
    ) -> Option<AssocSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        fn extract_node_id(t: &Ty) -> Option<NodeId> {
            match t.kind {
                TyKind::Path(None, _) => Some(t.id),
                TyKind::Ref(_, ref mut_ty) => extract_node_id(&mut_ty.ty),
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
                    if let Some(Res::Def(DefKind::Struct | DefKind::Union, did)) =
                        resolution.full_res()
                    {
                        if let Some(field_names) = self.r.field_names.get(&did) {
                            if field_names.iter().any(|&field_name| ident.name == field_name.node) {
                                return Some(AssocSuggestion::Field);
                            }
                        }
                    }
                }
            }
        }

        if let Some(items) = self.diagnostic_metadata.current_trait_assoc_items {
            for assoc_item in items {
                if assoc_item.ident == ident {
                    return Some(match &assoc_item.kind {
                        ast::AssocItemKind::Const(..) => AssocSuggestion::AssocConst,
                        ast::AssocItemKind::Fn(box ast::Fn { sig, .. }) if sig.decl.has_self() => {
                            AssocSuggestion::MethodWithSelf { called }
                        }
                        ast::AssocItemKind::Fn(..) => AssocSuggestion::AssocFn { called },
                        ast::AssocItemKind::Type(..) => AssocSuggestion::AssocType,
                        ast::AssocItemKind::MacCall(_) => continue,
                    });
                }
            }
        }

        // Look for associated items in the current trait.
        if let Some((module, _)) = self.current_trait_ref {
            if let Ok(binding) = self.r.maybe_resolve_ident_in_module(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                &self.parent_scope,
            ) {
                let res = binding.res();
                if filter_fn(res) {
                    if self.r.has_self.contains(&res.def_id()) {
                        return Some(AssocSuggestion::MethodWithSelf { called });
                    } else {
                        match res {
                            Res::Def(DefKind::AssocFn, _) => {
                                return Some(AssocSuggestion::AssocFn { called });
                            }
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
    ) -> TypoCandidate {
        let mut names = Vec::new();
        if path.len() == 1 {
            let mut ctxt = path.last().unwrap().ident.span.ctxt();

            // Search in lexical scope.
            // Walk backwards up the ribs in scope and collect candidates.
            for rib in self.ribs[ns].iter().rev() {
                let rib_ctxt = if rib.kind.contains_params() {
                    ctxt.normalize_to_macros_2_0()
                } else {
                    ctxt.normalize_to_macro_rules()
                };

                // Locals and type parameters
                for (ident, &res) in &rib.bindings {
                    if filter_fn(res) && ident.span.ctxt() == rib_ctxt {
                        names.push(TypoSuggestion::typo_from_ident(*ident, res));
                    }
                }

                if let RibKind::MacroDefinition(def) = rib.kind && def == self.r.macro_def(ctxt) {
                    // If an invocation of this macro created `ident`, give up on `ident`
                    // and switch to `ident`'s source from the macro definition.
                    ctxt.remove_mark();
                    continue;
                }

                // Items in scope
                if let RibKind::ModuleRibKind(module) = rib.kind {
                    // Items from this module
                    self.r.add_module_candidates(module, &mut names, &filter_fn, Some(ctxt));

                    if let ModuleKind::Block = module.kind {
                        // We can see through blocks
                    } else {
                        // Items from the prelude
                        if !module.no_implicit_prelude {
                            let extern_prelude = self.r.extern_prelude.clone();
                            names.extend(extern_prelude.iter().flat_map(|(ident, _)| {
                                self.r
                                    .crate_loader()
                                    .maybe_process_path_extern(ident.name)
                                    .and_then(|crate_id| {
                                        let crate_mod =
                                            Res::Def(DefKind::Mod, crate_id.as_def_id());

                                        filter_fn(crate_mod).then(|| {
                                            TypoSuggestion::typo_from_ident(*ident, crate_mod)
                                        })
                                    })
                            }));

                            if let Some(prelude) = self.r.prelude {
                                self.r.add_module_candidates(prelude, &mut names, &filter_fn, None);
                            }
                        }
                        break;
                    }
                }
            }
            // Add primitive types to the mix
            if filter_fn(Res::PrimTy(PrimTy::Bool)) {
                names.extend(PrimTy::ALL.iter().map(|prim_ty| {
                    TypoSuggestion::typo_from_name(prim_ty.name(), Res::PrimTy(*prim_ty))
                }))
            }
        } else {
            // Search in module.
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(mod_path, Some(TypeNS), None)
            {
                self.r.add_module_candidates(module, &mut names, &filter_fn, None);
            }
        }

        let name = path[path.len() - 1].ident.name;
        // Make sure error reporting is deterministic.
        names.sort_by(|a, b| a.candidate.as_str().partial_cmp(b.candidate.as_str()).unwrap());

        match find_best_match_for_name(
            &names.iter().map(|suggestion| suggestion.candidate).collect::<Vec<Symbol>>(),
            name,
            None,
        ) {
            Some(found) => {
                let Some(sugg) = names.into_iter().find(|suggestion| suggestion.candidate == found) else {
                    return TypoCandidate::None;
                };
                if found == name {
                    TypoCandidate::Shadowed(sugg.res, sugg.span)
                } else {
                    TypoCandidate::Typo(sugg)
                }
            }
            _ => TypoCandidate::None,
        }
    }

    // Returns the name of the Rust type approximately corresponding to
    // a type name in another programming language.
    fn likely_rust_type(path: &[Segment]) -> Option<Symbol> {
        let name = path[path.len() - 1].ident.as_str();
        // Common Java types
        Some(match name {
            "byte" => sym::u8, // In Java, bytes are signed, but in practice one almost always wants unsigned bytes.
            "short" => sym::i16,
            "Bool" => sym::bool,
            "Boolean" => sym::bool,
            "boolean" => sym::bool,
            "int" => sym::i32,
            "long" => sym::i64,
            "float" => sym::f32,
            "double" => sym::f64,
            _ => return None,
        })
    }

    /// Only used in a specific case of type ascription suggestions
    fn get_colon_suggestion_span(&self, start: Span) -> Span {
        let sm = self.r.session.source_map();
        start.to(sm.next_point(start))
    }

    fn type_ascription_suggestion(&self, err: &mut Diagnostic, base_span: Span) -> bool {
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
                            ";",
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
                                "::",
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
                                err.downgrade_to_delayed_bug();
                            }
                        }
                        if let Ok(base_snippet) = base_snippet {
                            // Try to find an assignment
                            let eq_span = sm.span_look_ahead(after_colon_sp, Some("="), Some(50));
                            if let Ok(ref snippet) = sm.span_to_snippet(eq_span) && snippet == "=" {
                                err.span_suggestion(
                                    base_span,
                                    "maybe you meant to write an assignment here",
                                    format!("let {}", base_snippet),
                                    Applicability::MaybeIncorrect,
                                );
                                show_label = false;
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

    // try to give a suggestion for this pattern: `name = blah`, which is common in other languages
    // suggest `let name = blah` to introduce a new binding
    fn let_binding_suggestion(&mut self, err: &mut Diagnostic, ident_span: Span) -> bool {
        if let Some(Expr { kind: ExprKind::Assign(lhs, .. ), .. }) = self.diagnostic_metadata.in_assignment &&
            let ast::ExprKind::Path(None, _) = lhs.kind {
                if !ident_span.from_expansion() {
                    err.span_suggestion_verbose(
                        ident_span.shrink_to_lo(),
                        "you might have meant to introduce a new binding",
                        "let ".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
            }
        false
    }

    fn find_module(&mut self, def_id: DefId) -> Option<(Module<'a>, ImportSuggestion)> {
        let mut result = None;
        let mut seen_modules = FxHashSet::default();
        let mut worklist = vec![(self.r.graph_root, ThinVec::new())];

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
                    let module_def_id = module.def_id();
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
                                note: None,
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
        err: &mut Diagnostic,
        source: PathSource<'_>,
        def_id: DefId,
        span: Span,
    ) {
        let Some(variants) = self.collect_enum_ctors(def_id) else {
            err.note("you might have meant to use one of the enum's variants");
            return;
        };

        let suggest_only_tuple_variants =
            matches!(source, PathSource::TupleStruct(..)) || source.is_call();
        if suggest_only_tuple_variants {
            // Suggest only tuple variants regardless of whether they have fields and do not
            // suggest path with added parentheses.
            let suggestable_variants = variants
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
                    suggestable_variants,
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
            let needs_placeholder = |ctor_def_id: DefId, kind: CtorKind| {
                let def_id = self.r.parent(ctor_def_id);
                let has_no_fields = self.r.field_names.get(&def_id).map_or(false, |f| f.is_empty());
                match kind {
                    CtorKind::Const => false,
                    CtorKind::Fn if has_no_fields => false,
                    _ => true,
                }
            };

            let suggestable_variants = variants
                .iter()
                .filter(|(_, def_id, kind)| !needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .map(|(variant, kind)| match kind {
                    CtorKind::Const => variant,
                    CtorKind::Fn => format!("({}())", variant),
                })
                .collect::<Vec<_>>();
            let no_suggestable_variant = suggestable_variants.is_empty();

            if !no_suggestable_variant {
                let msg = if suggestable_variants.len() == 1 {
                    "you might have meant to use the following enum variant"
                } else {
                    "you might have meant to use one of the following enum variants"
                };

                err.span_suggestions(
                    span,
                    msg,
                    suggestable_variants,
                    Applicability::MaybeIncorrect,
                );
            }

            let suggestable_variants_with_placeholders = variants
                .iter()
                .filter(|(_, def_id, kind)| needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .filter_map(|(variant, kind)| match kind {
                    CtorKind::Fn => Some(format!("({}(/* fields */))", variant)),
                    _ => None,
                })
                .collect::<Vec<_>>();

            if !suggestable_variants_with_placeholders.is_empty() {
                let msg =
                    match (no_suggestable_variant, suggestable_variants_with_placeholders.len()) {
                        (true, 1) => "the following enum variant is available",
                        (true, _) => "the following enum variants are available",
                        (false, 1) => "alternatively, the following enum variant is available",
                        (false, _) => {
                            "alternatively, the following enum variants are also available"
                        }
                    };

                err.span_suggestions(
                    span,
                    msg,
                    suggestable_variants_with_placeholders,
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

    pub(crate) fn report_missing_type_error(
        &self,
        path: &[Segment],
    ) -> Option<(Span, &'static str, String, Applicability)> {
        let (ident, span) = match path {
            [segment]
                if !segment.has_generic_args
                    && segment.ident.name != kw::SelfUpper
                    && segment.ident.name != kw::Dyn =>
            {
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
        match (self.diagnostic_metadata.current_item, single_uppercase_char, self.diagnostic_metadata.currently_processing_generics) {
            (Some(Item { kind: ItemKind::Fn(..), ident, .. }), _, _) if ident.name == sym::main => {
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
                true, _
            )
            // Without the 2nd `true`, we'd suggest `impl <T>` for `impl T` when a type `T` isn't found
            | (Some(Item { kind: kind @ ItemKind::Impl(..), .. }), true, true)
            | (Some(Item { kind, .. }), false, _) => {
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
                        } else if let GenericParam {
                            kind: GenericParamKind::Const { ty, kw_span: _, default  }, ..
                        } = param {
                            default.as_ref().map(|def| def.value.span).unwrap_or(ty.span)
                        } else {
                            param.ident.span
                        };
                        (span, format!(", {}", ident))
                    } else {
                        (generics.span, format!("<{}>", ident))
                    };
                    // Do not suggest if this is coming from macro expansion.
                    if span.can_be_used_for_suggestions() {
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
    pub(crate) fn suggestion_for_label_in_rib(
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
            .filter(|(id, _)| id.span.eq_ctxt(label.span))
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

    pub(crate) fn maybe_report_lifetime_uses(
        &mut self,
        generics_span: Span,
        params: &[ast::GenericParam],
    ) {
        for (param_index, param) in params.iter().enumerate() {
            let GenericParamKind::Lifetime = param.kind else { continue };

            let def_id = self.r.local_def_id(param.id);

            let use_set = self.lifetime_uses.remove(&def_id);
            debug!(
                "Use set for {:?}({:?} at {:?}) is {:?}",
                def_id, param.ident, param.ident.span, use_set
            );

            let deletion_span = || {
                if params.len() == 1 {
                    // if sole lifetime, remove the entire `<>` brackets
                    Some(generics_span)
                } else if param_index == 0 {
                    // if removing within `<>` brackets, we also want to
                    // delete a leading or trailing comma as appropriate
                    match (
                        param.span().find_ancestor_inside(generics_span),
                        params[param_index + 1].span().find_ancestor_inside(generics_span),
                    ) {
                        (Some(param_span), Some(next_param_span)) => {
                            Some(param_span.to(next_param_span.shrink_to_lo()))
                        }
                        _ => None,
                    }
                } else {
                    // if removing within `<>` brackets, we also want to
                    // delete a leading or trailing comma as appropriate
                    match (
                        param.span().find_ancestor_inside(generics_span),
                        params[param_index - 1].span().find_ancestor_inside(generics_span),
                    ) {
                        (Some(param_span), Some(prev_param_span)) => {
                            Some(prev_param_span.shrink_to_hi().to(param_span))
                        }
                        _ => None,
                    }
                }
            };
            match use_set {
                Some(LifetimeUseSet::Many) => {}
                Some(LifetimeUseSet::One { use_span, use_ctxt }) => {
                    debug!(?param.ident, ?param.ident.span, ?use_span);

                    let elidable = matches!(use_ctxt, LifetimeCtxt::Ref);

                    let deletion_span = deletion_span();
                    self.r.lint_buffer.buffer_lint_with_diagnostic(
                        lint::builtin::SINGLE_USE_LIFETIMES,
                        param.id,
                        param.ident.span,
                        &format!("lifetime parameter `{}` only used once", param.ident),
                        lint::BuiltinLintDiagnostics::SingleUseLifetime {
                            param_span: param.ident.span,
                            use_span: Some((use_span, elidable)),
                            deletion_span,
                        },
                    );
                }
                None => {
                    debug!(?param.ident, ?param.ident.span);
                    let deletion_span = deletion_span();
                    // the give lifetime originates from expanded code so we won't be able to remove it #104432
                    let lifetime_only_in_expanded_code =
                        deletion_span.map(|sp| sp.in_derive_expansion()).unwrap_or(true);
                    if !lifetime_only_in_expanded_code {
                        self.r.lint_buffer.buffer_lint_with_diagnostic(
                            lint::builtin::UNUSED_LIFETIMES,
                            param.id,
                            param.ident.span,
                            &format!("lifetime parameter `{}` never used", param.ident),
                            lint::BuiltinLintDiagnostics::SingleUseLifetime {
                                param_span: param.ident.span,
                                use_span: None,
                                deletion_span,
                            },
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn emit_undeclared_lifetime_error(
        &self,
        lifetime_ref: &ast::Lifetime,
        outer_lifetime_ref: Option<Ident>,
    ) {
        debug_assert_ne!(lifetime_ref.ident.name, kw::UnderscoreLifetime);
        let mut err = if let Some(outer) = outer_lifetime_ref {
            let mut err = struct_span_err!(
                self.r.session,
                lifetime_ref.ident.span,
                E0401,
                "can't use generic parameters from outer item",
            );
            err.span_label(lifetime_ref.ident.span, "use of generic parameter from outer item");
            err.span_label(outer.span, "lifetime parameter from outer item");
            err
        } else {
            let mut err = struct_span_err!(
                self.r.session,
                lifetime_ref.ident.span,
                E0261,
                "use of undeclared lifetime name `{}`",
                lifetime_ref.ident
            );
            err.span_label(lifetime_ref.ident.span, "undeclared lifetime");
            err
        };
        self.suggest_introducing_lifetime(
            &mut err,
            Some(lifetime_ref.ident.name.as_str()),
            |err, _, span, message, suggestion| {
                err.span_suggestion(span, message, suggestion, Applicability::MaybeIncorrect);
                true
            },
        );
        err.emit();
    }

    fn suggest_introducing_lifetime(
        &self,
        err: &mut Diagnostic,
        name: Option<&str>,
        suggest: impl Fn(&mut Diagnostic, bool, Span, &str, String) -> bool,
    ) {
        let mut suggest_note = true;
        for rib in self.lifetime_ribs.iter().rev() {
            let mut should_continue = true;
            match rib.kind {
                LifetimeRibKind::Generics { binder: _, span, kind } => {
                    if !span.can_be_used_for_suggestions() && suggest_note && let Some(name) = name {
                        suggest_note = false; // Avoid displaying the same help multiple times.
                        err.span_label(
                            span,
                            &format!(
                                "lifetime `{}` is missing in item created through this procedural macro",
                                name,
                            ),
                        );
                        continue;
                    }

                    let higher_ranked = matches!(
                        kind,
                        LifetimeBinderKind::BareFnType
                            | LifetimeBinderKind::PolyTrait
                            | LifetimeBinderKind::WhereBound
                    );
                    let (span, sugg) = if span.is_empty() {
                        let sugg = format!(
                            "{}<{}>{}",
                            if higher_ranked { "for" } else { "" },
                            name.unwrap_or("'a"),
                            if higher_ranked { " " } else { "" },
                        );
                        (span, sugg)
                    } else {
                        let span =
                            self.r.session.source_map().span_through_char(span, '<').shrink_to_hi();
                        let sugg = format!("{}, ", name.unwrap_or("'a"));
                        (span, sugg)
                    };
                    if higher_ranked {
                        let message = format!(
                            "consider making the {} lifetime-generic with a new `{}` lifetime",
                            kind.descr(),
                            name.unwrap_or("'a"),
                        );
                        should_continue = suggest(err, true, span, &message, sugg);
                        err.note_once(
                            "for more information on higher-ranked polymorphism, visit \
                             https://doc.rust-lang.org/nomicon/hrtb.html",
                        );
                    } else if let Some(name) = name {
                        let message = format!("consider introducing lifetime `{}` here", name);
                        should_continue = suggest(err, false, span, &message, sugg);
                    } else {
                        let message = "consider introducing a named lifetime parameter";
                        should_continue = suggest(err, false, span, &message, sugg);
                    }
                }
                LifetimeRibKind::Item => break,
                _ => {}
            }
            if !should_continue {
                break;
            }
        }
    }

    pub(crate) fn emit_non_static_lt_in_const_generic_error(&self, lifetime_ref: &ast::Lifetime) {
        struct_span_err!(
            self.r.session,
            lifetime_ref.ident.span,
            E0771,
            "use of non-static lifetime `{}` in const generic",
            lifetime_ref.ident
        )
        .note(
            "for more information, see issue #74052 \
            <https://github.com/rust-lang/rust/issues/74052>",
        )
        .emit();
    }

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `generic_const_exprs` is not enabled, the body identified by
    /// `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    pub(crate) fn maybe_emit_forbidden_non_static_lifetime_error(
        &self,
        lifetime_ref: &ast::Lifetime,
    ) {
        let feature_active = self.r.session.features_untracked().generic_const_exprs;
        if !feature_active {
            feature_err(
                &self.r.session.parse_sess,
                sym::generic_const_exprs,
                lifetime_ref.ident.span,
                "a non-static lifetime is not allowed in a `const`",
            )
            .emit();
        }
    }

    pub(crate) fn report_missing_lifetime_specifiers(
        &mut self,
        lifetime_refs: Vec<MissingLifetime>,
        function_param_lifetimes: Option<(Vec<MissingLifetime>, Vec<ElisionFnParameter>)>,
    ) -> ErrorGuaranteed {
        let num_lifetimes: usize = lifetime_refs.iter().map(|lt| lt.count).sum();
        let spans: Vec<_> = lifetime_refs.iter().map(|lt| lt.span).collect();

        let mut err = struct_span_err!(
            self.r.session,
            spans,
            E0106,
            "missing lifetime specifier{}",
            pluralize!(num_lifetimes)
        );
        self.add_missing_lifetime_specifiers_label(
            &mut err,
            lifetime_refs,
            function_param_lifetimes,
        );
        err.emit()
    }

    fn add_missing_lifetime_specifiers_label(
        &mut self,
        err: &mut Diagnostic,
        lifetime_refs: Vec<MissingLifetime>,
        function_param_lifetimes: Option<(Vec<MissingLifetime>, Vec<ElisionFnParameter>)>,
    ) {
        for &lt in &lifetime_refs {
            err.span_label(
                lt.span,
                format!(
                    "expected {} lifetime parameter{}",
                    if lt.count == 1 { "named".to_string() } else { lt.count.to_string() },
                    pluralize!(lt.count),
                ),
            );
        }

        let mut in_scope_lifetimes: Vec<_> = self
            .lifetime_ribs
            .iter()
            .rev()
            .take_while(|rib| !matches!(rib.kind, LifetimeRibKind::Item))
            .flat_map(|rib| rib.bindings.iter())
            .map(|(&ident, &res)| (ident, res))
            .filter(|(ident, _)| ident.name != kw::UnderscoreLifetime)
            .collect();
        debug!(?in_scope_lifetimes);

        debug!(?function_param_lifetimes);
        if let Some((param_lifetimes, params)) = &function_param_lifetimes {
            let elided_len = param_lifetimes.len();
            let num_params = params.len();

            let mut m = String::new();

            for (i, info) in params.iter().enumerate() {
                let ElisionFnParameter { ident, index, lifetime_count, span } = *info;
                debug_assert_ne!(lifetime_count, 0);

                err.span_label(span, "");

                if i != 0 {
                    if i + 1 < num_params {
                        m.push_str(", ");
                    } else if num_params == 2 {
                        m.push_str(" or ");
                    } else {
                        m.push_str(", or ");
                    }
                }

                let help_name = if let Some(ident) = ident {
                    format!("`{}`", ident)
                } else {
                    format!("argument {}", index + 1)
                };

                if lifetime_count == 1 {
                    m.push_str(&help_name[..])
                } else {
                    m.push_str(&format!("one of {}'s {} lifetimes", help_name, lifetime_count)[..])
                }
            }

            if num_params == 0 {
                err.help(
                    "this function's return type contains a borrowed value, \
                 but there is no value for it to be borrowed from",
                );
                if in_scope_lifetimes.is_empty() {
                    in_scope_lifetimes = vec![(
                        Ident::with_dummy_span(kw::StaticLifetime),
                        (DUMMY_NODE_ID, LifetimeRes::Static),
                    )];
                }
            } else if elided_len == 0 {
                err.help(
                    "this function's return type contains a borrowed value with \
                 an elided lifetime, but the lifetime cannot be derived from \
                 the arguments",
                );
                if in_scope_lifetimes.is_empty() {
                    in_scope_lifetimes = vec![(
                        Ident::with_dummy_span(kw::StaticLifetime),
                        (DUMMY_NODE_ID, LifetimeRes::Static),
                    )];
                }
            } else if num_params == 1 {
                err.help(&format!(
                    "this function's return type contains a borrowed value, \
                 but the signature does not say which {} it is borrowed from",
                    m
                ));
            } else {
                err.help(&format!(
                    "this function's return type contains a borrowed value, \
                 but the signature does not say whether it is borrowed from {}",
                    m
                ));
            }
        }

        let existing_name = match &in_scope_lifetimes[..] {
            [] => Symbol::intern("'a"),
            [(existing, _)] => existing.name,
            _ => Symbol::intern("'lifetime"),
        };

        let mut spans_suggs: Vec<_> = Vec::new();
        let build_sugg = |lt: MissingLifetime| match lt.kind {
            MissingLifetimeKind::Underscore => {
                debug_assert_eq!(lt.count, 1);
                (lt.span, existing_name.to_string())
            }
            MissingLifetimeKind::Ampersand => {
                debug_assert_eq!(lt.count, 1);
                (lt.span.shrink_to_hi(), format!("{} ", existing_name))
            }
            MissingLifetimeKind::Comma => {
                let sugg: String = std::iter::repeat([existing_name.as_str(), ", "])
                    .take(lt.count)
                    .flatten()
                    .collect();
                (lt.span.shrink_to_hi(), sugg)
            }
            MissingLifetimeKind::Brackets => {
                let sugg: String = std::iter::once("<")
                    .chain(
                        std::iter::repeat(existing_name.as_str()).take(lt.count).intersperse(", "),
                    )
                    .chain([">"])
                    .collect();
                (lt.span.shrink_to_hi(), sugg)
            }
        };
        for &lt in &lifetime_refs {
            spans_suggs.push(build_sugg(lt));
        }
        debug!(?spans_suggs);
        match in_scope_lifetimes.len() {
            0 => {
                if let Some((param_lifetimes, _)) = function_param_lifetimes {
                    for lt in param_lifetimes {
                        spans_suggs.push(build_sugg(lt))
                    }
                }
                self.suggest_introducing_lifetime(
                    err,
                    None,
                    |err, higher_ranked, span, message, intro_sugg| {
                        err.multipart_suggestion_verbose(
                            message,
                            std::iter::once((span, intro_sugg))
                                .chain(spans_suggs.iter().cloned())
                                .collect(),
                            Applicability::MaybeIncorrect,
                        );
                        higher_ranked
                    },
                );
            }
            1 => {
                err.multipart_suggestion_verbose(
                    &format!("consider using the `{}` lifetime", existing_name),
                    spans_suggs,
                    Applicability::MaybeIncorrect,
                );

                // Record as using the suggested resolution.
                let (_, (_, res)) = in_scope_lifetimes[0];
                for &lt in &lifetime_refs {
                    self.r.lifetimes_res_map.insert(lt.id, res);
                }
            }
            _ => {
                let lifetime_spans: Vec<_> =
                    in_scope_lifetimes.iter().map(|(ident, _)| ident.span).collect();
                err.span_note(lifetime_spans, "these named lifetimes are available to use");

                if spans_suggs.len() > 0 {
                    // This happens when we have `Foo<T>` where we point at the space before `T`,
                    // but this can be confusing so we give a suggestion with placeholders.
                    err.multipart_suggestion_verbose(
                        "consider using one of the available lifetimes here",
                        spans_suggs,
                        Applicability::HasPlaceholders,
                    );
                }
            }
        }
    }
}

/// Report lifetime/lifetime shadowing as an error.
pub(super) fn signal_lifetime_shadowing(sess: &Session, orig: Ident, shadower: Ident) {
    let mut err = struct_span_err!(
        sess,
        shadower.span,
        E0496,
        "lifetime name `{}` shadows a lifetime name that is already in scope",
        orig.name,
    );
    err.span_label(orig.span, "first declared here");
    err.span_label(shadower.span, format!("lifetime `{}` already in scope", orig.name));
    err.emit();
}

/// Shadowing involving a label is only a warning for historical reasons.
//FIXME: make this a proper lint.
pub(super) fn signal_label_shadowing(sess: &Session, orig: Span, shadower: Ident) {
    let name = shadower.name;
    let shadower = shadower.span;
    let mut err = sess.struct_span_warn(
        shadower,
        &format!("label name `{}` shadows a label name that is already in scope", name),
    );
    err.span_label(orig, "first declared here");
    err.span_label(shadower, format!("label `{}` already in scope", name));
    err.emit();
}
