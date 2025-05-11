// ignore-tidy-filelength

use std::borrow::Cow;
use std::iter;
use std::ops::Deref;

use rustc_ast::ptr::P;
use rustc_ast::visit::{FnCtxt, FnKind, LifetimeCtxt, Visitor, walk_ty};
use rustc_ast::{
    self as ast, AssocItemKind, DUMMY_NODE_ID, Expr, ExprKind, GenericParam, GenericParamKind,
    Item, ItemKind, MethodCall, NodeId, Path, PathSegment, Ty, TyKind,
};
use rustc_ast_pretty::pprust::where_bound_predicate_to_string;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, MultiSpan, SuggestionStyle, pluralize,
    struct_span_code_err,
};
use rustc_hir as hir;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_hir::{MissingLifetimeKind, PrimTy};
use rustc_middle::ty;
use rustc_session::{Session, lint};
use rustc_span::edit_distance::{edit_distance, find_best_match_for_name};
use rustc_span::edition::Edition;
use rustc_span::hygiene::MacroKind;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use thin_vec::ThinVec;
use tracing::debug;

use super::NoConstantGenericsReason;
use crate::diagnostics::{ImportSuggestion, LabelSuggestion, TypoSuggestion};
use crate::late::{
    AliasPossibility, LateResolutionVisitor, LifetimeBinderKind, LifetimeRes, LifetimeRibKind,
    LifetimeUseSet, QSelf, RibKind,
};
use crate::ty::fast_reject::SimplifiedType;
use crate::{
    Module, ModuleKind, ModuleOrUniformRoot, PathResult, PathSource, Segment, errors,
    path_names_to_string,
};

type Res = def::Res<ast::NodeId>;

/// A field or associated item from self type suggested in case of resolution failure.
enum AssocSuggestion {
    Field(Span),
    MethodWithSelf { called: bool },
    AssocFn { called: bool },
    AssocType,
    AssocConst,
}

impl AssocSuggestion {
    fn action(&self) -> &'static str {
        match self {
            AssocSuggestion::Field(_) => "use the available field",
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
    /// Used to overwrite the resolution with the suggestion, to avoid cascading errors.
    pub id: NodeId,
    /// As we cannot yet emit lints in this crate and have to buffer them instead,
    /// we need to associate each lint with some `NodeId`,
    /// however for some `MissingLifetime`s their `NodeId`s are "fake",
    /// in a sense that they are temporary and not get preserved down the line,
    /// which means that the lints for those nodes will not get emitted.
    /// To combat this, we can try to use some other `NodeId`s as a fallback option.
    pub id_for_lint: NodeId,
    /// Where to suggest adding the lifetime.
    pub span: Span,
    /// How the lifetime was introduced, to have the correct space and comma.
    pub kind: MissingLifetimeKind,
    /// Number of elided lifetimes, used for elision in path.
    pub count: usize,
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
    module: Option<DefId>,
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

impl<'ast, 'ra: 'ast, 'tcx> LateResolutionVisitor<'_, 'ast, 'ra, 'tcx> {
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
                    Res::Def(DefKind::TyParam, def_id) => {
                        Some((self.r.def_span(def_id), "found this type parameter"))
                    }
                    _ => None,
                },
                could_be_expr: match res {
                    Res::Def(DefKind::Fn, _) => {
                        // Verify whether this is a fn call or an Fn used as a type.
                        self.r
                            .tcx
                            .sess
                            .source_map()
                            .span_to_snippet(span)
                            .is_ok_and(|snippet| snippet.ends_with(')'))
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
                module: None,
            }
        } else {
            let mut span_label = None;
            let item_ident = path.last().unwrap().ident;
            let item_span = item_ident.span;
            let (mod_prefix, mod_str, module, suggestion) = if path.len() == 1 {
                debug!(?self.diag_metadata.current_impl_items);
                debug!(?self.diag_metadata.current_function);
                let suggestion = if self.current_trait_ref.is_none()
                    && let Some((fn_kind, _)) = self.diag_metadata.current_function
                    && let Some(FnCtxt::Assoc(_)) = fn_kind.ctxt()
                    && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = fn_kind
                    && let Some(items) = self.diag_metadata.current_impl_items
                    && let Some(item) = items.iter().find(|i| {
                        i.kind.ident().is_some_and(|ident| {
                            // Don't suggest if the item is in Fn signature arguments (#112590).
                            ident.name == item_str.name && !sig.span.contains(item_span)
                        })
                    }) {
                    let sp = item_span.shrink_to_lo();

                    // Account for `Foo { field }` when suggesting `self.field` so we result on
                    // `Foo { field: self.field }`.
                    let field = match source {
                        PathSource::Expr(Some(Expr { kind: ExprKind::Struct(expr), .. })) => {
                            expr.fields.iter().find(|f| f.ident == item_ident)
                        }
                        _ => None,
                    };
                    let pre = if let Some(field) = field
                        && field.is_shorthand
                    {
                        format!("{item_ident}: ")
                    } else {
                        String::new()
                    };
                    // Ensure we provide a structured suggestion for an assoc fn only for
                    // expressions that are actually a fn call.
                    let is_call = match field {
                        Some(ast::ExprField { expr, .. }) => {
                            matches!(expr.kind, ExprKind::Call(..))
                        }
                        _ => matches!(
                            source,
                            PathSource::Expr(Some(Expr { kind: ExprKind::Call(..), .. })),
                        ),
                    };

                    match &item.kind {
                        AssocItemKind::Fn(fn_)
                            if (!sig.decl.has_self() || !is_call) && fn_.sig.decl.has_self() =>
                        {
                            // Ensure that we only suggest `self.` if `self` is available,
                            // you can't call `fn foo(&self)` from `fn bar()` (#115992).
                            // We also want to mention that the method exists.
                            span_label = Some((
                                fn_.ident.span,
                                "a method by that name is available on `Self` here",
                            ));
                            None
                        }
                        AssocItemKind::Fn(fn_) if !fn_.sig.decl.has_self() && !is_call => {
                            span_label = Some((
                                fn_.ident.span,
                                "an associated function by that name is available on `Self` here",
                            ));
                            None
                        }
                        AssocItemKind::Fn(fn_) if fn_.sig.decl.has_self() => {
                            Some((sp, "consider using the method on `Self`", format!("{pre}self.")))
                        }
                        AssocItemKind::Fn(_) => Some((
                            sp,
                            "consider using the associated function on `Self`",
                            format!("{pre}Self::"),
                        )),
                        AssocItemKind::Const(..) => Some((
                            sp,
                            "consider using the associated constant on `Self`",
                            format!("{pre}Self::"),
                        )),
                        _ => None,
                    }
                } else {
                    None
                };
                (String::new(), "this scope".to_string(), None, suggestion)
            } else if path.len() == 2 && path[0].ident.name == kw::PathRoot {
                if self.r.tcx.sess.edition() > Edition::Edition2015 {
                    // In edition 2018 onwards, the `::foo` syntax may only pull from the extern prelude
                    // which overrides all other expectations of item type
                    expected = "crate";
                    (String::new(), "the list of imported crates".to_string(), None, None)
                } else {
                    (
                        String::new(),
                        "the crate root".to_string(),
                        Some(CRATE_DEF_ID.to_def_id()),
                        None,
                    )
                }
            } else if path.len() == 2 && path[0].ident.name == kw::Crate {
                (String::new(), "the crate root".to_string(), Some(CRATE_DEF_ID.to_def_id()), None)
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_res = self.resolve_path(mod_path, Some(TypeNS), None);
                let mod_prefix = match mod_res {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => module.res(),
                    _ => None,
                };

                let module_did = mod_prefix.as_ref().and_then(Res::mod_def_id);

                let mod_prefix =
                    mod_prefix.map_or_else(String::new, |res| (format!("{} ", res.descr())));

                (mod_prefix, format!("`{}`", Segment::names_to_string(mod_path)), module_did, None)
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
                span_label,
                could_be_expr: false,
                suggestion,
                module,
            }
        }
    }

    /// Try to suggest for a module path that cannot be resolved.
    /// Such as `fmt::Debug` where `fmt` is not resolved without importing,
    /// here we search with `lookup_import_candidates` for a module named `fmt`
    /// with `TypeNS` as namespace.
    ///
    /// We need a separate function here because we won't suggest for a path with single segment
    /// and we won't change `SourcePath` api `is_expected` to match `Type` with `DefKind::Mod`
    pub(crate) fn smart_resolve_partial_mod_path_errors(
        &mut self,
        prefix_path: &[Segment],
        following_seg: Option<&Segment>,
    ) -> Vec<ImportSuggestion> {
        if let Some(segment) = prefix_path.last()
            && let Some(following_seg) = following_seg
        {
            let candidates = self.r.lookup_import_candidates(
                segment.ident,
                Namespace::TypeNS,
                &self.parent_scope,
                &|res: Res| matches!(res, Res::Def(DefKind::Mod, _)),
            );
            // double check next seg is valid
            candidates
                .into_iter()
                .filter(|candidate| {
                    if let Some(def_id) = candidate.did
                        && let Some(module) = self.r.get_module(def_id)
                    {
                        Some(def_id) != self.parent_scope.module.opt_def_id()
                            && self
                                .r
                                .resolutions(module)
                                .borrow()
                                .iter()
                                .any(|(key, _r)| key.ident.name == following_seg.ident.name)
                    } else {
                        false
                    }
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    /// Handles error reporting for `smart_resolve_path_fragment` function.
    /// Creates base error and amends it with one short label and possibly some longer helps/notes.
    pub(crate) fn smart_resolve_report_errors(
        &mut self,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        source: PathSource<'_>,
        res: Option<Res>,
        qself: Option<&QSelf>,
    ) -> (Diag<'tcx>, Vec<ImportSuggestion>) {
        debug!(?res, ?source);
        let base_error = self.make_base_error(path, span, source, res);

        let code = source.error_code(res.is_some());
        let mut err = self.r.dcx().struct_span_err(base_error.span, base_error.msg.clone());
        err.code(code);

        // Try to get the span of the identifier within the path's syntax context
        // (if that's different).
        if let Some(within_macro_span) =
            base_error.span.within_macro(span, self.r.tcx.sess.source_map())
        {
            err.span_label(within_macro_span, "due to this macro variable");
        }

        self.detect_missing_binding_available_from_pattern(&mut err, path, following_seg);
        self.suggest_at_operator_in_slice_pat_with_range(&mut err, path);
        self.suggest_swapping_misplaced_self_ty_and_trait(&mut err, source, res, base_error.span);

        if let Some((span, label)) = base_error.span_label {
            err.span_label(span, label);
        }

        if let Some(ref sugg) = base_error.suggestion {
            err.span_suggestion_verbose(sugg.0, sugg.1, &sugg.2, Applicability::MaybeIncorrect);
        }

        self.suggest_changing_type_to_const_param(&mut err, res, source, span);
        self.explain_functions_in_pattern(&mut err, res, source);

        if self.suggest_pattern_match_with_let(&mut err, source, span) {
            // Fallback label.
            err.span_label(base_error.span, base_error.fallback_label);
            return (err, Vec::new());
        }

        self.suggest_self_or_self_ref(&mut err, path, span);
        self.detect_assoc_type_constraint_meant_as_path(&mut err, &base_error);
        self.detect_rtn_with_fully_qualified_path(
            &mut err,
            path,
            following_seg,
            span,
            source,
            res,
            qself,
        );
        if self.suggest_self_ty(&mut err, source, path, span)
            || self.suggest_self_value(&mut err, source, path, span)
        {
            return (err, Vec::new());
        }

        let (found, suggested_candidates, mut candidates) = self.try_lookup_name_relaxed(
            &mut err,
            source,
            path,
            following_seg,
            span,
            res,
            &base_error,
        );
        if found {
            return (err, candidates);
        }

        if self.suggest_shadowed(&mut err, source, path, following_seg, span) {
            // if there is already a shadowed name, don'suggest candidates for importing
            candidates.clear();
        }

        let mut fallback = self.suggest_trait_and_bounds(&mut err, source, res, span, &base_error);
        fallback |= self.suggest_typo(
            &mut err,
            source,
            path,
            following_seg,
            span,
            &base_error,
            suggested_candidates,
        );

        if fallback {
            // Fallback label.
            err.span_label(base_error.span, base_error.fallback_label);
        }
        self.err_code_special_cases(&mut err, source, path, span);

        if let Some(module) = base_error.module {
            self.r.find_cfg_stripped(&mut err, &path.last().unwrap().ident.name, module);
        }

        (err, candidates)
    }

    fn detect_rtn_with_fully_qualified_path(
        &self,
        err: &mut Diag<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        source: PathSource<'_>,
        res: Option<Res>,
        qself: Option<&QSelf>,
    ) {
        if let Some(Res::Def(DefKind::AssocFn, _)) = res
            && let PathSource::TraitItem(TypeNS) = source
            && let None = following_seg
            && let Some(qself) = qself
            && let TyKind::Path(None, ty_path) = &qself.ty.kind
            && ty_path.segments.len() == 1
            && self.diag_metadata.current_where_predicate.is_some()
        {
            err.span_suggestion_verbose(
                span,
                "you might have meant to use the return type notation syntax",
                format!("{}::{}(..)", ty_path.segments[0].ident, path[path.len() - 1].ident),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn detect_assoc_type_constraint_meant_as_path(
        &self,
        err: &mut Diag<'_>,
        base_error: &BaseError,
    ) {
        let Some(ty) = self.diag_metadata.current_type_path else {
            return;
        };
        let TyKind::Path(_, path) = &ty.kind else {
            return;
        };
        for segment in &path.segments {
            let Some(params) = &segment.args else {
                continue;
            };
            let ast::GenericArgs::AngleBracketed(params) = params.deref() else {
                continue;
            };
            for param in &params.args {
                let ast::AngleBracketedArg::Constraint(constraint) = param else {
                    continue;
                };
                let ast::AssocItemConstraintKind::Bound { bounds } = &constraint.kind else {
                    continue;
                };
                for bound in bounds {
                    let ast::GenericBound::Trait(trait_ref) = bound else {
                        continue;
                    };
                    if trait_ref.modifiers == ast::TraitBoundModifiers::NONE
                        && base_error.span == trait_ref.span
                    {
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

    fn suggest_self_or_self_ref(&mut self, err: &mut Diag<'_>, path: &[Segment], span: Span) {
        if !self.self_type_is_available() {
            return;
        }
        let Some(path_last_segment) = path.last() else { return };
        let item_str = path_last_segment.ident;
        // Emit help message for fake-self from other languages (e.g., `this` in JavaScript).
        if ["this", "my"].contains(&item_str.as_str()) {
            err.span_suggestion_short(
                span,
                "you might have meant to use `self` here instead",
                "self",
                Applicability::MaybeIncorrect,
            );
            if !self.self_value_is_available(path[0].ident.span) {
                if let Some((FnKind::Fn(_, _, ast::Fn { sig, .. }), fn_span)) =
                    &self.diag_metadata.current_function
                {
                    let (span, sugg) = if let Some(param) = sig.decl.inputs.get(0) {
                        (param.span.shrink_to_lo(), "&self, ")
                    } else {
                        (
                            self.r
                                .tcx
                                .sess
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
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        res: Option<Res>,
        base_error: &BaseError,
    ) -> (bool, FxHashSet<String>, Vec<ImportSuggestion>) {
        let span = match following_seg {
            Some(_) if path[0].ident.span.eq_ctxt(path[path.len() - 1].ident.span) => {
                // The path `span` that comes in includes any following segments, which we don't
                // want to replace in the suggestions.
                path[0].ident.span.to(path[path.len() - 1].ident.span)
            }
            _ => span,
        };
        let mut suggested_candidates = FxHashSet::default();
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
        // Try to filter out intrinsics candidates, as long as we have
        // some other candidates to suggest.
        let intrinsic_candidates: Vec<_> = candidates
            .extract_if(.., |sugg| {
                let path = path_names_to_string(&sugg.path);
                path.starts_with("core::intrinsics::") || path.starts_with("std::intrinsics::")
            })
            .collect();
        if candidates.is_empty() {
            // Put them back if we have no more candidates to suggest...
            candidates = intrinsic_candidates;
        }
        let crate_def_id = CRATE_DEF_ID.to_def_id();
        if candidates.is_empty() && is_expected(Res::Def(DefKind::Enum, crate_def_id)) {
            let mut enum_candidates: Vec<_> = self
                .r
                .lookup_import_candidates(ident, ns, &self.parent_scope, is_enum_variant)
                .into_iter()
                .map(|suggestion| import_candidate_to_enum_paths(&suggestion))
                .filter(|(_, enum_ty_path)| !enum_ty_path.starts_with("std::prelude::"))
                .collect();
            if !enum_candidates.is_empty() {
                enum_candidates.sort();

                // Contextualize for E0412 "cannot find type", but don't belabor the point
                // (that it's a variant) for E0573 "expected type, found variant".
                let preamble = if res.is_none() {
                    let others = match enum_candidates.len() {
                        1 => String::new(),
                        2 => " and 1 other".to_owned(),
                        n => format!(" and {n} others"),
                    };
                    format!("there is an enum variant `{}`{}; ", enum_candidates[0].0, others)
                } else {
                    String::new()
                };
                let msg = format!("{preamble}try using the variant's enum");

                suggested_candidates.extend(
                    enum_candidates
                        .iter()
                        .map(|(_variant_path, enum_ty_path)| enum_ty_path.clone()),
                );
                err.span_suggestions(
                    span,
                    msg,
                    enum_candidates.into_iter().map(|(_variant_path, enum_ty_path)| enum_ty_path),
                    Applicability::MachineApplicable,
                );
            }
        }

        // Try finding a suitable replacement.
        let typo_sugg = self
            .lookup_typo_candidate(path, following_seg, source.namespace(), is_expected)
            .to_opt_suggestion()
            .filter(|sugg| !suggested_candidates.contains(sugg.candidate.as_str()));
        if let [segment] = path
            && !matches!(source, PathSource::Delegation)
            && self.self_type_is_available()
        {
            if let Some(candidate) =
                self.lookup_assoc_candidate(ident, ns, is_expected, source.is_call())
            {
                let self_is_available = self.self_value_is_available(segment.ident.span);
                // Account for `Foo { field }` when suggesting `self.field` so we result on
                // `Foo { field: self.field }`.
                let pre = match source {
                    PathSource::Expr(Some(Expr { kind: ExprKind::Struct(expr), .. }))
                        if expr
                            .fields
                            .iter()
                            .any(|f| f.ident == segment.ident && f.is_shorthand) =>
                    {
                        format!("{path_str}: ")
                    }
                    _ => String::new(),
                };
                match candidate {
                    AssocSuggestion::Field(field_span) => {
                        if self_is_available {
                            err.span_suggestion_verbose(
                                span.shrink_to_lo(),
                                "you might have meant to use the available field",
                                format!("{pre}self."),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_label(field_span, "a field by that name exists in `Self`");
                        }
                    }
                    AssocSuggestion::MethodWithSelf { called } if self_is_available => {
                        let msg = if called {
                            "you might have meant to call the method"
                        } else {
                            "you might have meant to refer to the method"
                        };
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            msg,
                            "self.",
                            Applicability::MachineApplicable,
                        );
                    }
                    AssocSuggestion::MethodWithSelf { .. }
                    | AssocSuggestion::AssocFn { .. }
                    | AssocSuggestion::AssocConst
                    | AssocSuggestion::AssocType => {
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            format!("you might have meant to {}", candidate.action()),
                            "Self::",
                            Applicability::MachineApplicable,
                        );
                    }
                }
                self.r.add_typo_suggestion(err, typo_sugg, ident_span);
                return (true, suggested_candidates, candidates);
            }

            // If the first argument in call is `self` suggest calling a method.
            if let Some((call_span, args_span)) = self.call_has_self_arg(source) {
                let mut args_snippet = String::new();
                if let Some(args_span) = args_span {
                    if let Ok(snippet) = self.r.tcx.sess.source_map().span_to_snippet(args_span) {
                        args_snippet = snippet;
                    }
                }

                err.span_suggestion(
                    call_span,
                    format!("try calling `{ident}` as a method"),
                    format!("self.{path_str}({args_snippet})"),
                    Applicability::MachineApplicable,
                );
                return (true, suggested_candidates, candidates);
            }
        }

        // Try context-dependent help if relaxed lookup didn't work.
        if let Some(res) = res {
            if self.smart_resolve_context_dependent_help(
                err,
                span,
                source,
                path,
                res,
                &path_str,
                &base_error.fallback_label,
            ) {
                // We do this to avoid losing a secondary span when we override the main error span.
                self.r.add_typo_suggestion(err, typo_sugg, ident_span);
                return (true, suggested_candidates, candidates);
            }
        }

        // Try to find in last block rib
        if let Some(rib) = &self.last_block_rib
            && let RibKind::Normal = rib.kind
        {
            for (ident, &res) in &rib.bindings {
                if let Res::Local(_) = res
                    && path.len() == 1
                    && ident.span.eq_ctxt(path[0].ident.span)
                    && ident.name == path[0].ident.name
                {
                    err.span_help(
                        ident.span,
                        format!("the binding `{path_str}` is available in a different scope in the same function"),
                    );
                    return (true, suggested_candidates, candidates);
                }
            }
        }

        if candidates.is_empty() {
            candidates = self.smart_resolve_partial_mod_path_errors(path, following_seg);
        }

        (false, suggested_candidates, candidates)
    }

    fn suggest_trait_and_bounds(
        &mut self,
        err: &mut Diag<'_>,
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
            if let Some(bounds @ [first_bound, .., last_bound]) =
                self.diag_metadata.current_trait_object
            {
                fallback = true;
                let spans: Vec<Span> = bounds
                    .iter()
                    .map(|bound| bound.span())
                    .filter(|&sp| sp != base_error.span)
                    .collect();

                let start_span = first_bound.span();
                // `end_span` is the end of the poly trait ref (Foo + 'baz + Bar><)
                let end_span = last_bound.span();
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
                    ast::GenericBound::Outlives(_) | ast::GenericBound::Use(..) => true,
                    ast::GenericBound::Trait(tr) => tr.span == base_error.span,
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
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        base_error: &BaseError,
        suggested_candidates: FxHashSet<String>,
    ) -> bool {
        let is_expected = &|res| source.is_expected(res);
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let typo_sugg =
            self.lookup_typo_candidate(path, following_seg, source.namespace(), is_expected);
        let mut fallback = false;
        let typo_sugg = typo_sugg
            .to_opt_suggestion()
            .filter(|sugg| !suggested_candidates.contains(sugg.candidate.as_str()));
        if !self.r.add_typo_suggestion(err, typo_sugg, ident_span) {
            fallback = true;
            match self.diag_metadata.current_let_binding {
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

            // If the trait has a single item (which wasn't matched by the algorithm), suggest it
            let suggestion = self.get_single_associated_item(path, &source, is_expected);
            self.r.add_typo_suggestion(err, suggestion, ident_span);
        }

        if self.let_binding_suggestion(err, ident_span) {
            fallback = false;
        }

        fallback
    }

    fn suggest_shadowed(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
    ) -> bool {
        let is_expected = &|res| source.is_expected(res);
        let typo_sugg =
            self.lookup_typo_candidate(path, following_seg, source.namespace(), is_expected);
        let is_in_same_file = &|sp1, sp2| {
            let source_map = self.r.tcx.sess.source_map();
            let file1 = source_map.span_to_filename(sp1);
            let file2 = source_map.span_to_filename(sp2);
            file1 == file2
        };
        // print 'you might have meant' if the candidate is (1) is a shadowed name with
        // accessible definition and (2) either defined in the same crate as the typo
        // (could be in a different file) or introduced in the same file as the typo
        // (could belong to a different crate)
        if let TypoCandidate::Shadowed(res, Some(sugg_span)) = typo_sugg
            && res.opt_def_id().is_some_and(|id| id.is_local() || is_in_same_file(span, sugg_span))
        {
            err.span_label(
                sugg_span,
                format!("you might have meant to refer to this {}", res.descr()),
            );
            return true;
        }
        false
    }

    fn err_code_special_cases(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) {
        if let Some(err_code) = err.code {
            if err_code == E0425 {
                for label_rib in &self.label_ribs {
                    for (label_ident, node_id) in &label_rib.bindings {
                        let ident = path.last().unwrap().ident;
                        if format!("'{ident}") == label_ident.to_string() {
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
                                self.diag_metadata.unused_labels.swap_remove(node_id);
                            }
                        }
                    }
                }
            } else if err_code == E0412 {
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
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) -> bool {
        if !is_self_type(path, source.namespace()) {
            return false;
        }
        err.code(E0411);
        err.span_label(span, "`Self` is only available in impls, traits, and type definitions");
        if let Some(item) = self.diag_metadata.current_item {
            if let Some(ident) = item.kind.ident() {
                err.span_label(
                    ident.span,
                    format!("`Self` not allowed in {} {}", item.kind.article(), item.kind.descr()),
                );
            }
        }
        true
    }

    fn suggest_self_value(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        path: &[Segment],
        span: Span,
    ) -> bool {
        if !is_self_value(path, source.namespace()) {
            return false;
        }

        debug!("smart_resolve_path_fragment: E0424, source={:?}", source);
        err.code(E0424);
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
        let self_from_macro = "a `self` parameter, but a macro invocation can only \
                               access identifiers it receives from parameters";
        if let Some((fn_kind, span)) = &self.diag_metadata.current_function {
            // The current function has a `self` parameter, but we were unable to resolve
            // a reference to `self`. This can only happen if the `self` identifier we
            // are resolving came from a different hygiene context.
            if fn_kind.decl().inputs.get(0).is_some_and(|p| p.is_self()) {
                err.span_label(*span, format!("this function has {self_from_macro}"));
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
                                    .tcx
                                    .sess
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
                        format!("this function {doesnt} have a `self` parameter"),
                    );
                }
            }
        } else if let Some(item) = self.diag_metadata.current_item {
            if matches!(item.kind, ItemKind::Delegation(..)) {
                err.span_label(item.span, format!("delegation supports {self_from_macro}"));
            } else {
                let span = if let Some(ident) = item.kind.ident() { ident.span } else { item.span };
                err.span_label(
                    span,
                    format!("`self` not allowed in {} {}", item.kind.article(), item.kind.descr()),
                );
            }
        }
        true
    }

    fn detect_missing_binding_available_from_pattern(
        &mut self,
        err: &mut Diag<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
    ) {
        let [segment] = path else { return };
        let None = following_seg else { return };
        for rib in self.ribs[ValueNS].iter().rev() {
            let patterns_with_skipped_bindings = self.r.tcx.with_stable_hashing_context(|hcx| {
                rib.patterns_with_skipped_bindings.to_sorted(&hcx, true)
            });
            for (def_id, spans) in patterns_with_skipped_bindings {
                if let DefKind::Struct | DefKind::Variant = self.r.tcx.def_kind(*def_id)
                    && let Some(fields) = self.r.field_idents(*def_id)
                {
                    for field in fields {
                        if field.name == segment.ident.name {
                            if spans.iter().all(|(_, had_error)| had_error.is_err()) {
                                // This resolution error will likely be fixed by fixing a
                                // syntax error in a pattern, so it is irrelevant to the user.
                                let multispan: MultiSpan =
                                    spans.iter().map(|(s, _)| *s).collect::<Vec<_>>().into();
                                err.span_note(
                                    multispan,
                                    "this pattern had a recovered parse error which likely lost \
                                     the expected fields",
                                );
                                err.downgrade_to_delayed_bug();
                            }
                            let ty = self.r.tcx.item_name(*def_id);
                            for (span, _) in spans {
                                err.span_label(
                                    *span,
                                    format!(
                                        "this pattern doesn't include `{field}`, which is \
                                         available in `{ty}`",
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn suggest_at_operator_in_slice_pat_with_range(
        &mut self,
        err: &mut Diag<'_>,
        path: &[Segment],
    ) {
        let Some(pat) = self.diag_metadata.current_pat else { return };
        let (bound, side, range) = match &pat.kind {
            ast::PatKind::Range(Some(bound), None, range) => (bound, Side::Start, range),
            ast::PatKind::Range(None, Some(bound), range) => (bound, Side::End, range),
            _ => return,
        };
        if let ExprKind::Path(None, range_path) = &bound.kind
            && let [segment] = &range_path.segments[..]
            && let [s] = path
            && segment.ident == s.ident
            && segment.ident.span.eq_ctxt(range.span)
        {
            // We've encountered `[first, rest..]` (#88404) or `[first, ..rest]` (#120591)
            // where the user might have meant `[first, rest @ ..]`.
            let (span, snippet) = match side {
                Side::Start => (segment.ident.span.between(range.span), " @ ".into()),
                Side::End => (range.span.to(segment.ident.span), format!("{} @ ..", segment.ident)),
            };
            err.subdiagnostic(errors::UnexpectedResUseAtOpInSlicePatWithRangeSugg {
                span,
                ident: segment.ident,
                snippet,
            });
        }

        enum Side {
            Start,
            End,
        }
    }

    fn suggest_swapping_misplaced_self_ty_and_trait(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        res: Option<Res>,
        span: Span,
    ) {
        if let Some((trait_ref, self_ty)) =
            self.diag_metadata.currently_processing_impl_trait.clone()
            && let TyKind::Path(_, self_ty_path) = &self_ty.kind
            && let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(&Segment::from_path(self_ty_path), Some(TypeNS), None)
            && let ModuleKind::Def(DefKind::Trait, ..) = module.kind
            && trait_ref.path.span == span
            && let PathSource::Trait(_) = source
            && let Some(Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _)) = res
            && let Ok(self_ty_str) = self.r.tcx.sess.source_map().span_to_snippet(self_ty.span)
            && let Ok(trait_ref_str) =
                self.r.tcx.sess.source_map().span_to_snippet(trait_ref.path.span)
        {
            err.multipart_suggestion(
                    "`impl` items mention the trait being implemented first and the type it is being implemented for second",
                    vec![(trait_ref.path.span, self_ty_str), (self_ty.span, trait_ref_str)],
                    Applicability::MaybeIncorrect,
                );
        }
    }

    fn explain_functions_in_pattern(
        &mut self,
        err: &mut Diag<'_>,
        res: Option<Res>,
        source: PathSource<'_>,
    ) {
        let PathSource::TupleStruct(_, _) = source else { return };
        let Some(Res::Def(DefKind::Fn, _)) = res else { return };
        err.primary_message("expected a pattern, found a function call");
        err.note("function calls are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>");
    }

    fn suggest_changing_type_to_const_param(
        &mut self,
        err: &mut Diag<'_>,
        res: Option<Res>,
        source: PathSource<'_>,
        span: Span,
    ) {
        let PathSource::Trait(_) = source else { return };

        // We don't include `DefKind::Str` and `DefKind::AssocTy` as they can't be reached here anyway.
        let applicability = match res {
            Some(Res::PrimTy(PrimTy::Int(_) | PrimTy::Uint(_) | PrimTy::Bool | PrimTy::Char)) => {
                Applicability::MachineApplicable
            }
            // FIXME(const_generics): Add `DefKind::TyParam` and `SelfTyParam` once we support generic
            // const generics. Of course, `Struct` and `Enum` may contain ty params, too, but the
            // benefits of including them here outweighs the small number of false positives.
            Some(Res::Def(DefKind::Struct | DefKind::Enum, _))
                if self.r.tcx.features().adt_const_params() =>
            {
                Applicability::MaybeIncorrect
            }
            _ => return,
        };

        let Some(item) = self.diag_metadata.current_item else { return };
        let Some(generics) = item.kind.generics() else { return };

        let param = generics.params.iter().find_map(|param| {
            // Only consider type params with exactly one trait bound.
            if let [bound] = &*param.bounds
                && let ast::GenericBound::Trait(tref) = bound
                && tref.modifiers == ast::TraitBoundModifiers::NONE
                && tref.span == span
                && param.ident.span.eq_ctxt(span)
            {
                Some(param.ident.span)
            } else {
                None
            }
        });

        if let Some(param) = param {
            err.subdiagnostic(errors::UnexpectedResChangeTyToConstParamSugg {
                span: param.shrink_to_lo(),
                applicability,
            });
        }
    }

    fn suggest_pattern_match_with_let(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        span: Span,
    ) -> bool {
        if let PathSource::Expr(_) = source
            && let Some(Expr { span: expr_span, kind: ExprKind::Assign(lhs, _, _), .. }) =
                self.diag_metadata.in_if_condition
        {
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
                if let [target] = targets.as_slice() {
                    return Some(TypoSuggestion::single_item_from_ident(target.0.ident, target.1));
                }
            }
        }
        None
    }

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    fn restrict_assoc_type_in_where_clause(&mut self, span: Span, err: &mut Diag<'_>) -> bool {
        // Detect that we are actually in a `where` predicate.
        let (bounded_ty, bounds, where_span) = if let Some(ast::WherePredicate {
            kind:
                ast::WherePredicateKind::BoundPredicate(ast::WhereBoundPredicate {
                    bounded_ty,
                    bound_generic_params,
                    bounds,
                }),
            span,
            ..
        }) = self.diag_metadata.current_where_predicate
        {
            if !bound_generic_params.is_empty() {
                return false;
            }
            (bounded_ty, bounds, span)
        } else {
            return false;
        };

        // Confirm that the target is an associated type.
        let (ty, _, path) = if let ast::TyKind::Path(Some(qself), path) = &bounded_ty.kind {
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
                [ast::PathSegment { args: None, .. }],
                [ast::GenericBound::Trait(poly_trait_ref)],
            ) = (&type_param_path.segments[..], &bounds[..])
                && poly_trait_ref.modifiers == ast::TraitBoundModifiers::NONE
            {
                if let [ast::PathSegment { ident, args: None, .. }] =
                    &poly_trait_ref.trait_ref.path.segments[..]
                {
                    if ident.span == span {
                        let Some(new_where_bound_predicate) =
                            mk_where_bound_predicate(path, poly_trait_ref, ty)
                        else {
                            return false;
                        };
                        err.span_suggestion_verbose(
                            *where_span,
                            format!("constrain the associated type to `{ident}`"),
                            where_bound_predicate_to_string(&new_where_bound_predicate),
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
        if let PathSource::Expr(Some(parent)) = source
            && let ExprKind::Call(_, args) = &parent.kind
            && !args.is_empty()
        {
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
        has_self_arg
    }

    fn followed_by_brace(&self, span: Span) -> (bool, Option<Span>) {
        // HACK(estebank): find a better way to figure out that this was a
        // parser issue where a struct literal is being used on an expression
        // where a brace being opened means a block is being started. Look
        // ahead for the next text to see if `span` is followed by a `{`.
        let sm = self.r.tcx.sess.source_map();
        if let Some(followed_brace_span) = sm.span_look_ahead(span, "{", Some(50)) {
            // In case this could be a struct literal that needs to be surrounded
            // by parentheses, find the appropriate span.
            let close_brace_span = sm.span_look_ahead(followed_brace_span, "}", Some(50));
            let closing_brace = close_brace_span.map(|sp| span.to(sp));
            (true, closing_brace)
        } else {
            (false, None)
        }
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut Diag<'_>,
        span: Span,
        source: PathSource<'_>,
        path: &[Segment],
        res: Res,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);

        let path_sep = |this: &mut Self, err: &mut Diag<'_>, expr: &Expr, kind: DefKind| {
            const MESSAGE: &str = "use the path separator to refer to an item";

            let (lhs_span, rhs_span) = match &expr.kind {
                ExprKind::Field(base, ident) => (base.span, ident.span),
                ExprKind::MethodCall(box MethodCall { receiver, span, .. }) => {
                    (receiver.span, *span)
                }
                _ => return false,
            };

            if lhs_span.eq_ctxt(rhs_span) {
                err.span_suggestion_verbose(
                    lhs_span.between(rhs_span),
                    MESSAGE,
                    "::",
                    Applicability::MaybeIncorrect,
                );
                true
            } else if matches!(kind, DefKind::Struct | DefKind::TyAlias)
                && let Some(lhs_source_span) = lhs_span.find_ancestor_inside(expr.span)
                && let Ok(snippet) = this.r.tcx.sess.source_map().span_to_snippet(lhs_source_span)
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

        let find_span = |source: &PathSource<'_>, err: &mut Diag<'_>| {
            match source {
                PathSource::Expr(Some(Expr { span, kind: ExprKind::Call(_, _), .. }))
                | PathSource::TupleStruct(span, _) => {
                    // We want the main underline to cover the suggested code as well for
                    // cleaner output.
                    err.span(*span);
                    *span
                }
                _ => span,
            }
        };

        let bad_struct_syntax_suggestion = |this: &mut Self, err: &mut Diag<'_>, def_id: DefId| {
            let (followed_by_brace, closing_brace) = this.followed_by_brace(span);

            match source {
                PathSource::Expr(Some(
                    parent @ Expr { kind: ExprKind::Field(..) | ExprKind::MethodCall(..), .. },
                )) if path_sep(this, err, parent, DefKind::Struct) => {}
                PathSource::Expr(
                    None
                    | Some(Expr {
                        kind:
                            ExprKind::Path(..)
                            | ExprKind::Binary(..)
                            | ExprKind::Unary(..)
                            | ExprKind::If(..)
                            | ExprKind::While(..)
                            | ExprKind::ForLoop { .. }
                            | ExprKind::Match(..),
                        ..
                    }),
                ) if followed_by_brace => {
                    if let Some(sp) = closing_brace {
                        err.span_label(span, fallback_label.to_string());
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
                                 `({path_str} {{ /* fields */ }})`?"
                            ),
                        );
                    }
                }
                PathSource::Expr(_) | PathSource::TupleStruct(..) | PathSource::Pat => {
                    let span = find_span(&source, err);
                    err.span_label(this.r.def_span(def_id), format!("`{path_str}` defined here"));

                    let (tail, descr, applicability, old_fields) = match source {
                        PathSource::Pat => ("", "pattern", Applicability::MachineApplicable, None),
                        PathSource::TupleStruct(_, args) => (
                            "",
                            "pattern",
                            Applicability::MachineApplicable,
                            Some(
                                args.iter()
                                    .map(|a| this.r.tcx.sess.source_map().span_to_snippet(*a).ok())
                                    .collect::<Vec<Option<String>>>(),
                            ),
                        ),
                        _ => (": val", "literal", Applicability::HasPlaceholders, None),
                    };

                    if !this.has_private_fields(def_id) {
                        // If the fields of the type are private, we shouldn't be suggesting using
                        // the struct literal syntax at all, as that will cause a subsequent error.
                        let fields = this.r.field_idents(def_id);
                        let has_fields = fields.as_ref().is_some_and(|f| !f.is_empty());

                        if let PathSource::Expr(Some(Expr {
                            kind: ExprKind::Call(path, args),
                            span,
                            ..
                        })) = source
                            && !args.is_empty()
                            && let Some(fields) = &fields
                            && args.len() == fields.len()
                        // Make sure we have same number of args as fields
                        {
                            let path_span = path.span;
                            let mut parts = Vec::new();

                            // Start with the opening brace
                            parts.push((
                                path_span.shrink_to_hi().until(args[0].span),
                                "{".to_owned(),
                            ));

                            for (field, arg) in fields.iter().zip(args.iter()) {
                                // Add the field name before the argument
                                parts.push((arg.span.shrink_to_lo(), format!("{}: ", field)));
                            }

                            // Add the closing brace
                            parts.push((
                                args.last().unwrap().span.shrink_to_hi().until(span.shrink_to_hi()),
                                "}".to_owned(),
                            ));

                            err.multipart_suggestion_verbose(
                                format!("use struct {descr} syntax instead of calling"),
                                parts,
                                applicability,
                            );
                        } else {
                            let (fields, applicability) = match fields {
                                Some(fields) => {
                                    let fields = if let Some(old_fields) = old_fields {
                                        fields
                                            .iter()
                                            .enumerate()
                                            .map(|(idx, new)| (new, old_fields.get(idx)))
                                            .map(|(new, old)| {
                                                if let Some(Some(old)) = old
                                                    && new.as_str() != old
                                                {
                                                    format!("{new}: {old}")
                                                } else {
                                                    new.to_string()
                                                }
                                            })
                                            .collect::<Vec<String>>()
                                    } else {
                                        fields
                                            .iter()
                                            .map(|f| format!("{f}{tail}"))
                                            .collect::<Vec<String>>()
                                    };

                                    (fields.join(", "), applicability)
                                }
                                None => {
                                    ("/* fields */".to_string(), Applicability::HasPlaceholders)
                                }
                            };
                            let pad = if has_fields { " " } else { "" };
                            err.span_suggestion(
                                span,
                                format!("use struct {descr} syntax instead"),
                                format!("{path_str} {{{pad}{fields}{pad}}}"),
                                applicability,
                            );
                        }
                    }
                    if let PathSource::Expr(Some(Expr {
                        kind: ExprKind::Call(path, args),
                        span: call_span,
                        ..
                    })) = source
                    {
                        this.suggest_alternative_construction_methods(
                            def_id,
                            err,
                            path.span,
                            *call_span,
                            &args[..],
                        );
                    }
                }
                _ => {
                    err.span_label(span, fallback_label.to_string());
                }
            }
        };

        match (res, source) {
            (
                Res::Def(DefKind::Macro(MacroKind::Bang), def_id),
                PathSource::Expr(Some(Expr {
                    kind: ExprKind::Index(..) | ExprKind::Call(..), ..
                }))
                | PathSource::Struct,
            ) => {
                // Don't suggest macro if it's unstable.
                let suggestable = def_id.is_local()
                    || self.r.tcx.lookup_stability(def_id).is_none_or(|s| s.is_stable());

                err.span_label(span, fallback_label.to_string());

                // Don't suggest `!` for a macro invocation if there are generic args
                if path
                    .last()
                    .is_some_and(|segment| !segment.has_generic_args && !segment.has_lifetime_args)
                    && suggestable
                {
                    err.span_suggestion_verbose(
                        span.shrink_to_hi(),
                        "use `!` to invoke the macro",
                        "!",
                        Applicability::MaybeIncorrect,
                    );
                }

                if path_str == "try" && span.is_rust_2015() {
                    err.note("if you want the `try` keyword, you need Rust 2018 or later");
                }
            }
            (Res::Def(DefKind::Macro(MacroKind::Bang), _), _) => {
                err.span_label(span, fallback_label.to_string());
            }
            (Res::Def(DefKind::TyAlias, def_id), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if self.r.tcx.sess.is_nightly_build() {
                    let msg = "you might have meant to use `#![feature(trait_alias)]` instead of a \
                               `type` alias";
                    let span = self.r.def_span(def_id);
                    if let Ok(snip) = self.r.tcx.sess.source_map().span_to_snippet(span) {
                        // The span contains a type alias so we should be able to
                        // replace `type` with `trait`.
                        let snip = snip.replacen("type", "trait", 1);
                        err.span_suggestion(span, msg, snip, Applicability::MaybeIncorrect);
                    } else {
                        err.span_help(span, msg);
                    }
                }
            }
            (
                Res::Def(kind @ (DefKind::Mod | DefKind::Trait | DefKind::TyAlias), _),
                PathSource::Expr(Some(parent)),
            ) if path_sep(self, err, parent, kind) => {
                return true;
            }
            (
                Res::Def(DefKind::Enum, def_id),
                PathSource::TupleStruct(..) | PathSource::Expr(..),
            ) => {
                self.suggest_using_enum_variant(err, source, def_id, span);
            }
            (Res::Def(DefKind::Struct, def_id), source) if ns == ValueNS => {
                let struct_ctor = match def_id.as_local() {
                    Some(def_id) => self.r.struct_constructors.get(&def_id).cloned(),
                    None => {
                        let ctor = self.r.cstore().ctor_untracked(def_id);
                        ctor.map(|(ctor_kind, ctor_def_id)| {
                            let ctor_res =
                                Res::Def(DefKind::Ctor(CtorOf::Struct, ctor_kind), ctor_def_id);
                            let ctor_vis = self.r.tcx.visibility(ctor_def_id);
                            let field_visibilities = self
                                .r
                                .tcx
                                .associated_item_def_ids(def_id)
                                .iter()
                                .map(|field_id| self.r.tcx.visibility(field_id))
                                .collect();
                            (ctor_res, ctor_vis, field_visibilities)
                        })
                    }
                };

                let (ctor_def, ctor_vis, fields) = if let Some(struct_ctor) = struct_ctor {
                    if let PathSource::Expr(Some(parent)) = source {
                        if let ExprKind::Field(..) | ExprKind::MethodCall(..) = parent.kind {
                            bad_struct_syntax_suggestion(self, err, def_id);
                            return true;
                        }
                    }
                    struct_ctor
                } else {
                    bad_struct_syntax_suggestion(self, err, def_id);
                    return true;
                };

                let is_accessible = self.r.is_accessible_from(ctor_vis, self.parent_scope.module);
                if !is_expected(ctor_def) || is_accessible {
                    return true;
                }

                let field_spans = match source {
                    // e.g. `if let Enum::TupleVariant(field1, field2) = _`
                    PathSource::TupleStruct(_, pattern_spans) => {
                        err.primary_message(
                            "cannot match against a tuple struct which contains private fields",
                        );

                        // Use spans of the tuple struct pattern.
                        Some(Vec::from(pattern_spans))
                    }
                    // e.g. `let _ = Enum::TupleVariant(field1, field2);`
                    PathSource::Expr(Some(Expr {
                        kind: ExprKind::Call(path, args),
                        span: call_span,
                        ..
                    })) => {
                        err.primary_message(
                            "cannot initialize a tuple struct which contains private fields",
                        );
                        self.suggest_alternative_construction_methods(
                            def_id,
                            err,
                            path.span,
                            *call_span,
                            &args[..],
                        );
                        // Use spans of the tuple struct definition.
                        self.r
                            .field_idents(def_id)
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
                                format!(
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
                bad_struct_syntax_suggestion(self, err, def_id);
            }
            (Res::Def(DefKind::Ctor(_, CtorKind::Const), def_id), _) if ns == ValueNS => {
                match source {
                    PathSource::Expr(_) | PathSource::TupleStruct(..) | PathSource::Pat => {
                        let span = find_span(&source, err);
                        err.span_label(
                            self.r.def_span(def_id),
                            format!("`{path_str}` defined here"),
                        );
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
                let def_id = self.r.tcx.parent(ctor_def_id);
                err.span_label(self.r.def_span(def_id), format!("`{path_str}` defined here"));
                let fields = self.r.field_idents(def_id).map_or_else(
                    || "/* fields */".to_string(),
                    |field_ids| vec!["_"; field_ids.len()].join(", "),
                );
                err.span_suggestion(
                    span,
                    "use the tuple variant pattern syntax instead",
                    format!("{path_str}({fields})"),
                    Applicability::HasPlaceholders,
                );
            }
            (Res::SelfTyParam { .. } | Res::SelfTyAlias { .. }, _) if ns == ValueNS => {
                err.span_label(span, fallback_label.to_string());
                err.note("can't use `Self` as a constructor, you must use the implemented struct");
            }
            (Res::Def(DefKind::TyAlias | DefKind::AssocTy, _), _) if ns == ValueNS => {
                err.note("can't use a type alias as a constructor");
            }
            _ => return false,
        }
        true
    }

    fn suggest_alternative_construction_methods(
        &mut self,
        def_id: DefId,
        err: &mut Diag<'_>,
        path_span: Span,
        call_span: Span,
        args: &[P<Expr>],
    ) {
        if def_id.is_local() {
            // Doing analysis on local `DefId`s would cause infinite recursion.
            return;
        }
        // Look at all the associated functions without receivers in the type's
        // inherent impls to look for builders that return `Self`
        let mut items = self
            .r
            .tcx
            .inherent_impls(def_id)
            .iter()
            .flat_map(|i| self.r.tcx.associated_items(i).in_definition_order())
            // Only assoc fn with no receivers.
            .filter(|item| item.is_fn() && !item.is_method())
            .filter_map(|item| {
                // Only assoc fns that return `Self`
                let fn_sig = self.r.tcx.fn_sig(item.def_id).skip_binder();
                // Don't normalize the return type, because that can cause cycle errors.
                let ret_ty = fn_sig.output().skip_binder();
                let ty::Adt(def, _args) = ret_ty.kind() else {
                    return None;
                };
                let input_len = fn_sig.inputs().skip_binder().len();
                if def.did() != def_id {
                    return None;
                }
                let name = item.name();
                let order = !name.as_str().starts_with("new");
                Some((order, name, input_len))
            })
            .collect::<Vec<_>>();
        items.sort_by_key(|(order, _, _)| *order);
        let suggestion = |name, args| {
            format!(
                "::{name}({})",
                std::iter::repeat("_").take(args).collect::<Vec<_>>().join(", ")
            )
        };
        match &items[..] {
            [] => {}
            [(_, name, len)] if *len == args.len() => {
                err.span_suggestion_verbose(
                    path_span.shrink_to_hi(),
                    format!("you might have meant to use the `{name}` associated function",),
                    format!("::{name}"),
                    Applicability::MaybeIncorrect,
                );
            }
            [(_, name, len)] => {
                err.span_suggestion_verbose(
                    path_span.shrink_to_hi().with_hi(call_span.hi()),
                    format!("you might have meant to use the `{name}` associated function",),
                    suggestion(name, *len),
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {
                err.span_suggestions_with_style(
                    path_span.shrink_to_hi().with_hi(call_span.hi()),
                    "you might have meant to use an associated function to build this type",
                    items.iter().map(|(_, name, len)| suggestion(name, *len)),
                    Applicability::MaybeIncorrect,
                    SuggestionStyle::ShowAlways,
                );
            }
        }
        // We'd ideally use `type_implements_trait` but don't have access to
        // the trait solver here. We can't use `get_diagnostic_item` or
        // `all_traits` in resolve either. So instead we abuse the import
        // suggestion machinery to get `std::default::Default` and perform some
        // checks to confirm that we got *only* that trait. We then see if the
        // Adt we have has a direct implementation of `Default`. If so, we
        // provide a structured suggestion.
        let default_trait = self
            .r
            .lookup_import_candidates(
                Ident::with_dummy_span(sym::Default),
                Namespace::TypeNS,
                &self.parent_scope,
                &|res: Res| matches!(res, Res::Def(DefKind::Trait, _)),
            )
            .iter()
            .filter_map(|candidate| candidate.did)
            .find(|did| {
                self.r
                    .tcx
                    .get_attrs(*did, sym::rustc_diagnostic_item)
                    .any(|attr| attr.value_str() == Some(sym::Default))
            });
        let Some(default_trait) = default_trait else {
            return;
        };
        if self
            .r
            .extern_crate_map
            .items()
            // FIXME: This doesn't include impls like `impl Default for String`.
            .flat_map(|(_, crate_)| self.r.tcx.implementations_of_trait((*crate_, default_trait)))
            .filter_map(|(_, simplified_self_ty)| *simplified_self_ty)
            .filter_map(|simplified_self_ty| match simplified_self_ty {
                SimplifiedType::Adt(did) => Some(did),
                _ => None,
            })
            .any(|did| did == def_id)
        {
            err.multipart_suggestion(
                "consider using the `Default` trait",
                vec![
                    (path_span.shrink_to_lo(), "<".to_string()),
                    (
                        path_span.shrink_to_hi().with_hi(call_span.hi()),
                        " as std::default::Default>::default()".to_string(),
                    ),
                ],
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn has_private_fields(&self, def_id: DefId) -> bool {
        let fields = match def_id.as_local() {
            Some(def_id) => self.r.struct_constructors.get(&def_id).cloned().map(|(_, _, f)| f),
            None => Some(
                self.r
                    .tcx
                    .associated_item_def_ids(def_id)
                    .iter()
                    .map(|field_id| self.r.tcx.visibility(field_id))
                    .collect(),
            ),
        };

        fields.is_some_and(|fields| {
            fields.iter().any(|vis| !self.r.is_accessible_from(*vis, self.parent_scope.module))
        })
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

        let resolutions = self.r.resolutions(*module);
        let targets = resolutions
            .borrow()
            .iter()
            .filter_map(|(key, res)| res.borrow().binding.map(|binding| (key, binding.res())))
            .filter(|(_, res)| match (kind, res) {
                (AssocItemKind::Const(..), Res::Def(DefKind::AssocConst, _)) => true,
                (AssocItemKind::Fn(_), Res::Def(DefKind::AssocFn, _)) => true,
                (AssocItemKind::Type(..), Res::Def(DefKind::AssocTy, _)) => true,
                (AssocItemKind::Delegation(_), Res::Def(DefKind::AssocFn, _)) => true,
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
                self.diag_metadata.current_self_type.as_ref().and_then(extract_node_id)
            {
                // Look for a field with the same name in the current self_type.
                if let Some(resolution) = self.r.partial_res_map.get(&node_id) {
                    if let Some(Res::Def(DefKind::Struct | DefKind::Union, did)) =
                        resolution.full_res()
                    {
                        if let Some(fields) = self.r.field_idents(did) {
                            if let Some(field) = fields.iter().find(|id| ident.name == id.name) {
                                return Some(AssocSuggestion::Field(field.span));
                            }
                        }
                    }
                }
            }
        }

        if let Some(items) = self.diag_metadata.current_trait_assoc_items {
            for assoc_item in items {
                if let Some(assoc_ident) = assoc_item.kind.ident()
                    && assoc_ident == ident
                {
                    return Some(match &assoc_item.kind {
                        ast::AssocItemKind::Const(..) => AssocSuggestion::AssocConst,
                        ast::AssocItemKind::Fn(box ast::Fn { sig, .. }) if sig.decl.has_self() => {
                            AssocSuggestion::MethodWithSelf { called }
                        }
                        ast::AssocItemKind::Fn(..) => AssocSuggestion::AssocFn { called },
                        ast::AssocItemKind::Type(..) => AssocSuggestion::AssocType,
                        ast::AssocItemKind::Delegation(..)
                            if self
                                .r
                                .delegation_fn_sigs
                                .get(&self.r.local_def_id(assoc_item.id))
                                .is_some_and(|sig| sig.has_self) =>
                        {
                            AssocSuggestion::MethodWithSelf { called }
                        }
                        ast::AssocItemKind::Delegation(..) => AssocSuggestion::AssocFn { called },
                        ast::AssocItemKind::MacCall(_) | ast::AssocItemKind::DelegationMac(..) => {
                            continue;
                        }
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
                None,
            ) {
                let res = binding.res();
                if filter_fn(res) {
                    match res {
                        Res::Def(DefKind::Fn | DefKind::AssocFn, def_id) => {
                            let has_self = match def_id.as_local() {
                                Some(def_id) => self
                                    .r
                                    .delegation_fn_sigs
                                    .get(&def_id)
                                    .is_some_and(|sig| sig.has_self),
                                None => {
                                    self.r.tcx.fn_arg_idents(def_id).first().is_some_and(|&ident| {
                                        matches!(ident, Some(Ident { name: kw::SelfLower, .. }))
                                    })
                                }
                            };
                            if has_self {
                                return Some(AssocSuggestion::MethodWithSelf { called });
                            } else {
                                return Some(AssocSuggestion::AssocFn { called });
                            }
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

        None
    }

    fn lookup_typo_candidate(
        &mut self,
        path: &[Segment],
        following_seg: Option<&Segment>,
        ns: Namespace,
        filter_fn: &impl Fn(Res) -> bool,
    ) -> TypoCandidate {
        let mut names = Vec::new();
        if let [segment] = path {
            let mut ctxt = segment.ident.span.ctxt();

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

                if let RibKind::MacroDefinition(def) = rib.kind
                    && def == self.r.macro_def(ctxt)
                {
                    // If an invocation of this macro created `ident`, give up on `ident`
                    // and switch to `ident`'s source from the macro definition.
                    ctxt.remove_mark();
                    continue;
                }

                // Items in scope
                if let RibKind::Module(module) = rib.kind {
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
                                    .crate_loader(|c| c.maybe_process_path_extern(ident.name))
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

        // if next_seg is present, let's filter everything that does not continue the path
        if let Some(following_seg) = following_seg {
            names.retain(|suggestion| match suggestion.res {
                Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _) => {
                    // FIXME: this is not totally accurate, but mostly works
                    suggestion.candidate != following_seg.ident.name
                }
                Res::Def(DefKind::Mod, def_id) => self.r.get_module(def_id).map_or_else(
                    || false,
                    |module| {
                        self.r
                            .resolutions(module)
                            .borrow()
                            .iter()
                            .any(|(key, _)| key.ident.name == following_seg.ident.name)
                    },
                ),
                _ => true,
            });
        }
        let name = path[path.len() - 1].ident.name;
        // Make sure error reporting is deterministic.
        names.sort_by(|a, b| a.candidate.as_str().cmp(b.candidate.as_str()));

        match find_best_match_for_name(
            &names.iter().map(|suggestion| suggestion.candidate).collect::<Vec<Symbol>>(),
            name,
            None,
        ) {
            Some(found) => {
                let Some(sugg) = names.into_iter().find(|suggestion| suggestion.candidate == found)
                else {
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

    // try to give a suggestion for this pattern: `name = blah`, which is common in other languages
    // suggest `let name = blah` to introduce a new binding
    fn let_binding_suggestion(&mut self, err: &mut Diag<'_>, ident_span: Span) -> bool {
        if ident_span.from_expansion() {
            return false;
        }

        // only suggest when the code is a assignment without prefix code
        if let Some(Expr { kind: ExprKind::Assign(lhs, ..), .. }) = self.diag_metadata.in_assignment
            && let ast::ExprKind::Path(None, ref path) = lhs.kind
            && self.r.tcx.sess.source_map().is_line_before_span_empty(ident_span)
        {
            let (span, text) = match path.segments.first() {
                Some(seg) if let Some(name) = seg.ident.as_str().strip_prefix("let") => {
                    // a special case for #117894
                    let name = name.strip_prefix('_').unwrap_or(name);
                    (ident_span, format!("let {name}"))
                }
                _ => (ident_span.shrink_to_lo(), "let ".to_string()),
            };

            err.span_suggestion_verbose(
                span,
                "you might have meant to introduce a new binding",
                text,
                Applicability::MaybeIncorrect,
            );
            return true;
        }

        // a special case for #133713
        // '=' maybe a typo of `:`, which is a type annotation instead of assignment
        if err.code == Some(E0423)
            && let Some((let_span, None, Some(val_span))) = self.diag_metadata.current_let_binding
            && val_span.contains(ident_span)
            && val_span.lo() == ident_span.lo()
        {
            err.span_suggestion_verbose(
                let_span.shrink_to_hi().to(val_span.shrink_to_lo()),
                "you might have meant to use `:` for type annotation",
                ": ",
                Applicability::MaybeIncorrect,
            );
            return true;
        }
        false
    }

    fn find_module(&mut self, def_id: DefId) -> Option<(Module<'ra>, ImportSuggestion)> {
        let mut result = None;
        let mut seen_modules = FxHashSet::default();
        let root_did = self.r.graph_root.def_id();
        let mut worklist = vec![(
            self.r.graph_root,
            ThinVec::new(),
            root_did.is_local() || !self.r.tcx.is_doc_hidden(root_did),
        )];

        while let Some((in_module, path_segments, doc_visible)) = worklist.pop() {
            // abort if the module is already found
            if result.is_some() {
                break;
            }

            in_module.for_each_child(self.r, |r, ident, _, name_binding| {
                // abort if the module is already found or if name_binding is private external
                if result.is_some() || !name_binding.vis.is_visible_locally() {
                    return;
                }
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));
                    let module_def_id = module.def_id();
                    let doc_visible = doc_visible
                        && (module_def_id.is_local() || !r.tcx.is_doc_hidden(module_def_id));
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
                                doc_visible,
                                note: None,
                                via_import: false,
                                is_stable: true,
                            },
                        ));
                    } else {
                        // add the module to the lookup
                        if seen_modules.insert(module_def_id) {
                            worklist.push((module, path_segments, doc_visible));
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
        err: &mut Diag<'_>,
        source: PathSource<'_>,
        def_id: DefId,
        span: Span,
    ) {
        let Some(variant_ctors) = self.collect_enum_ctors(def_id) else {
            err.note("you might have meant to use one of the enum's variants");
            return;
        };

        // If the expression is a field-access or method-call, try to find a variant with the field/method name
        // that could have been intended, and suggest replacing the `.` with `::`.
        // Otherwise, suggest adding `::VariantName` after the enum;
        // and if the expression is call-like, only suggest tuple variants.
        let (suggest_path_sep_dot_span, suggest_only_tuple_variants) = match source {
            // `Type(a, b)` in a pattern, only suggest adding a tuple variant after `Type`.
            PathSource::TupleStruct(..) => (None, true),
            PathSource::Expr(Some(expr)) => match &expr.kind {
                // `Type(a, b)`, only suggest adding a tuple variant after `Type`.
                ExprKind::Call(..) => (None, true),
                // `Type.Foo(a, b)`, suggest replacing `.` -> `::` if variant `Foo` exists and is a tuple variant,
                // otherwise suggest adding a variant after `Type`.
                ExprKind::MethodCall(box MethodCall {
                    receiver,
                    span,
                    seg: PathSegment { ident, .. },
                    ..
                }) => {
                    let dot_span = receiver.span.between(*span);
                    let found_tuple_variant = variant_ctors.iter().any(|(path, _, ctor_kind)| {
                        *ctor_kind == CtorKind::Fn
                            && path.segments.last().is_some_and(|seg| seg.ident == *ident)
                    });
                    (found_tuple_variant.then_some(dot_span), false)
                }
                // `Type.Foo`, suggest replacing `.` -> `::` if variant `Foo` exists and is a unit or tuple variant,
                // otherwise suggest adding a variant after `Type`.
                ExprKind::Field(base, ident) => {
                    let dot_span = base.span.between(ident.span);
                    let found_tuple_or_unit_variant = variant_ctors.iter().any(|(path, ..)| {
                        path.segments.last().is_some_and(|seg| seg.ident == *ident)
                    });
                    (found_tuple_or_unit_variant.then_some(dot_span), false)
                }
                _ => (None, false),
            },
            _ => (None, false),
        };

        if let Some(dot_span) = suggest_path_sep_dot_span {
            err.span_suggestion_verbose(
                dot_span,
                "use the path separator to refer to a variant",
                "::",
                Applicability::MaybeIncorrect,
            );
        } else if suggest_only_tuple_variants {
            // Suggest only tuple variants regardless of whether they have fields and do not
            // suggest path with added parentheses.
            let mut suggestable_variants = variant_ctors
                .iter()
                .filter(|(.., kind)| *kind == CtorKind::Fn)
                .map(|(variant, ..)| path_names_to_string(variant))
                .collect::<Vec<_>>();
            suggestable_variants.sort();

            let non_suggestable_variant_count = variant_ctors.len() - suggestable_variants.len();

            let source_msg = if matches!(source, PathSource::TupleStruct(..)) {
                "to match against"
            } else {
                "to construct"
            };

            if !suggestable_variants.is_empty() {
                let msg = if non_suggestable_variant_count == 0 && suggestable_variants.len() == 1 {
                    format!("try {source_msg} the enum's variant")
                } else {
                    format!("try {source_msg} one of the enum's variants")
                };

                err.span_suggestions(
                    span,
                    msg,
                    suggestable_variants,
                    Applicability::MaybeIncorrect,
                );
            }

            // If the enum has no tuple variants..
            if non_suggestable_variant_count == variant_ctors.len() {
                err.help(format!("the enum has no tuple variants {source_msg}"));
            }

            // If there are also non-tuple variants..
            if non_suggestable_variant_count == 1 {
                err.help(format!("you might have meant {source_msg} the enum's non-tuple variant"));
            } else if non_suggestable_variant_count >= 1 {
                err.help(format!(
                    "you might have meant {source_msg} one of the enum's non-tuple variants"
                ));
            }
        } else {
            let needs_placeholder = |ctor_def_id: DefId, kind: CtorKind| {
                let def_id = self.r.tcx.parent(ctor_def_id);
                match kind {
                    CtorKind::Const => false,
                    CtorKind::Fn => {
                        !self.r.field_idents(def_id).is_some_and(|field_ids| field_ids.is_empty())
                    }
                }
            };

            let mut suggestable_variants = variant_ctors
                .iter()
                .filter(|(_, def_id, kind)| !needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .map(|(variant, kind)| match kind {
                    CtorKind::Const => variant,
                    CtorKind::Fn => format!("({variant}())"),
                })
                .collect::<Vec<_>>();
            suggestable_variants.sort();
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

            let mut suggestable_variants_with_placeholders = variant_ctors
                .iter()
                .filter(|(_, def_id, kind)| needs_placeholder(*def_id, *kind))
                .map(|(variant, _, kind)| (path_names_to_string(variant), kind))
                .filter_map(|(variant, kind)| match kind {
                    CtorKind::Fn => Some(format!("({variant}(/* fields */))")),
                    _ => None,
                })
                .collect::<Vec<_>>();
            suggestable_variants_with_placeholders.sort();

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
            err.span_note(self.r.def_span(def_id), "the enum is defined here");
        }
    }

    pub(crate) fn suggest_adding_generic_parameter(
        &self,
        path: &[Segment],
        source: PathSource<'_>,
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
        if !self.diag_metadata.currently_processing_generic_args && !single_uppercase_char {
            return None;
        }
        match (self.diag_metadata.current_item, single_uppercase_char, self.diag_metadata.currently_processing_generic_args) {
            (Some(Item { kind: ItemKind::Fn(fn_), .. }), _, _) if fn_.ident.name == sym::main => {
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

                    let (msg, sugg) = match source {
                        PathSource::Type | PathSource::PreciseCapturingArg(TypeNS) => {
                            ("you might be missing a type parameter", ident)
                        }
                        PathSource::Expr(_) | PathSource::PreciseCapturingArg(ValueNS) => (
                            "you might be missing a const parameter",
                            format!("const {ident}: /* Type */"),
                        ),
                        _ => return None,
                    };
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
                        (span, format!(", {sugg}"))
                    } else {
                        (generics.span, format!("<{sugg}>"))
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
                    let deletion_span =
                        if param.bounds.is_empty() { deletion_span() } else { None };

                    self.r.lint_buffer.buffer_lint(
                        lint::builtin::SINGLE_USE_LIFETIMES,
                        param.id,
                        param.ident.span,
                        lint::BuiltinLintDiag::SingleUseLifetime {
                            param_span: param.ident.span,
                            use_span: Some((use_span, elidable)),
                            deletion_span,
                            ident: param.ident,
                        },
                    );
                }
                None => {
                    debug!(?param.ident, ?param.ident.span);
                    let deletion_span = deletion_span();

                    // if the lifetime originates from expanded code, we won't be able to remove it #104432
                    if deletion_span.is_some_and(|sp| !sp.in_derive_expansion()) {
                        self.r.lint_buffer.buffer_lint(
                            lint::builtin::UNUSED_LIFETIMES,
                            param.id,
                            param.ident.span,
                            lint::BuiltinLintDiag::SingleUseLifetime {
                                param_span: param.ident.span,
                                use_span: None,
                                deletion_span,
                                ident: param.ident,
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
            struct_span_code_err!(
                self.r.dcx(),
                lifetime_ref.ident.span,
                E0401,
                "can't use generic parameters from outer item",
            )
            .with_span_label(lifetime_ref.ident.span, "use of generic parameter from outer item")
            .with_span_label(outer.span, "lifetime parameter from outer item")
        } else {
            struct_span_code_err!(
                self.r.dcx(),
                lifetime_ref.ident.span,
                E0261,
                "use of undeclared lifetime name `{}`",
                lifetime_ref.ident
            )
            .with_span_label(lifetime_ref.ident.span, "undeclared lifetime")
        };

        // Check if this is a typo of `'static`.
        if edit_distance(lifetime_ref.ident.name.as_str(), "'static", 2).is_some() {
            err.span_suggestion_verbose(
                lifetime_ref.ident.span,
                "you may have misspelled the `'static` lifetime",
                "'static",
                Applicability::MachineApplicable,
            );
        } else {
            self.suggest_introducing_lifetime(
                &mut err,
                Some(lifetime_ref.ident.name.as_str()),
                |err, _, span, message, suggestion, span_suggs| {
                    err.multipart_suggestion_with_style(
                        message,
                        std::iter::once((span, suggestion)).chain(span_suggs.clone()).collect(),
                        Applicability::MaybeIncorrect,
                        if span_suggs.is_empty() {
                            SuggestionStyle::ShowCode
                        } else {
                            SuggestionStyle::ShowAlways
                        },
                    );
                    true
                },
            );
        }

        err.emit();
    }

    fn suggest_introducing_lifetime(
        &self,
        err: &mut Diag<'_>,
        name: Option<&str>,
        suggest: impl Fn(
            &mut Diag<'_>,
            bool,
            Span,
            Cow<'static, str>,
            String,
            Vec<(Span, String)>,
        ) -> bool,
    ) {
        let mut suggest_note = true;
        for rib in self.lifetime_ribs.iter().rev() {
            let mut should_continue = true;
            match rib.kind {
                LifetimeRibKind::Generics { binder, span, kind } => {
                    // Avoid suggesting placing lifetime parameters on constant items unless the relevant
                    // feature is enabled. Suggest the parent item as a possible location if applicable.
                    if let LifetimeBinderKind::ConstItem = kind
                        && !self.r.tcx().features().generic_const_items()
                    {
                        continue;
                    }

                    if !span.can_be_used_for_suggestions()
                        && suggest_note
                        && let Some(name) = name
                    {
                        suggest_note = false; // Avoid displaying the same help multiple times.
                        err.span_label(
                            span,
                            format!(
                                "lifetime `{name}` is missing in item created through this procedural macro",
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

                    let mut rm_inner_binders: FxIndexSet<Span> = Default::default();
                    let (span, sugg) = if span.is_empty() {
                        let mut binder_idents: FxIndexSet<Ident> = Default::default();
                        binder_idents.insert(Ident::from_str(name.unwrap_or("'a")));

                        // We need to special case binders in the following situation:
                        // Change `T: for<'a> Trait<T> + 'b` to `for<'a, 'b> T: Trait<T> + 'b`
                        // T: for<'a> Trait<T> + 'b
                        //    ^^^^^^^  remove existing inner binder `for<'a>`
                        // for<'a, 'b> T: Trait<T> + 'b
                        // ^^^^^^^^^^^  suggest outer binder `for<'a, 'b>`
                        if let LifetimeBinderKind::WhereBound = kind
                            && let Some(predicate) = self.diag_metadata.current_where_predicate
                            && let ast::WherePredicateKind::BoundPredicate(
                                ast::WhereBoundPredicate { bounded_ty, bounds, .. },
                            ) = &predicate.kind
                            && bounded_ty.id == binder
                        {
                            for bound in bounds {
                                if let ast::GenericBound::Trait(poly_trait_ref) = bound
                                    && let span = poly_trait_ref
                                        .span
                                        .with_hi(poly_trait_ref.trait_ref.path.span.lo())
                                    && !span.is_empty()
                                {
                                    rm_inner_binders.insert(span);
                                    poly_trait_ref.bound_generic_params.iter().for_each(|v| {
                                        binder_idents.insert(v.ident);
                                    });
                                }
                            }
                        }

                        let binders_sugg = binder_idents.into_iter().enumerate().fold(
                            "".to_string(),
                            |mut binders, (i, x)| {
                                if i != 0 {
                                    binders += ", ";
                                }
                                binders += x.as_str();
                                binders
                            },
                        );
                        let sugg = format!(
                            "{}<{}>{}",
                            if higher_ranked { "for" } else { "" },
                            binders_sugg,
                            if higher_ranked { " " } else { "" },
                        );
                        (span, sugg)
                    } else {
                        let span = self
                            .r
                            .tcx
                            .sess
                            .source_map()
                            .span_through_char(span, '<')
                            .shrink_to_hi();
                        let sugg = format!("{}, ", name.unwrap_or("'a"));
                        (span, sugg)
                    };

                    if higher_ranked {
                        let message = Cow::from(format!(
                            "consider making the {} lifetime-generic with a new `{}` lifetime",
                            kind.descr(),
                            name.unwrap_or("'a"),
                        ));
                        should_continue = suggest(
                            err,
                            true,
                            span,
                            message,
                            sugg,
                            if !rm_inner_binders.is_empty() {
                                rm_inner_binders
                                    .into_iter()
                                    .map(|v| (v, "".to_string()))
                                    .collect::<Vec<_>>()
                            } else {
                                vec![]
                            },
                        );
                        err.note_once(
                            "for more information on higher-ranked polymorphism, visit \
                             https://doc.rust-lang.org/nomicon/hrtb.html",
                        );
                    } else if let Some(name) = name {
                        let message =
                            Cow::from(format!("consider introducing lifetime `{name}` here"));
                        should_continue = suggest(err, false, span, message, sugg, vec![]);
                    } else {
                        let message = Cow::from("consider introducing a named lifetime parameter");
                        should_continue = suggest(err, false, span, message, sugg, vec![]);
                    }
                }
                LifetimeRibKind::Item | LifetimeRibKind::ConstParamTy => break,
                _ => {}
            }
            if !should_continue {
                break;
            }
        }
    }

    pub(crate) fn emit_non_static_lt_in_const_param_ty_error(&self, lifetime_ref: &ast::Lifetime) {
        self.r
            .dcx()
            .create_err(errors::ParamInTyOfConstParam {
                span: lifetime_ref.ident.span,
                name: lifetime_ref.ident.name,
            })
            .emit();
    }

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `generic_const_exprs` is not enabled, the body identified by
    /// `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    pub(crate) fn emit_forbidden_non_static_lifetime_error(
        &self,
        cause: NoConstantGenericsReason,
        lifetime_ref: &ast::Lifetime,
    ) {
        match cause {
            NoConstantGenericsReason::IsEnumDiscriminant => {
                self.r
                    .dcx()
                    .create_err(errors::ParamInEnumDiscriminant {
                        span: lifetime_ref.ident.span,
                        name: lifetime_ref.ident.name,
                        param_kind: errors::ParamKindInEnumDiscriminant::Lifetime,
                    })
                    .emit();
            }
            NoConstantGenericsReason::NonTrivialConstArg => {
                assert!(!self.r.tcx.features().generic_const_exprs());
                self.r
                    .dcx()
                    .create_err(errors::ParamInNonTrivialAnonConst {
                        span: lifetime_ref.ident.span,
                        name: lifetime_ref.ident.name,
                        param_kind: errors::ParamKindInNonTrivialAnonConst::Lifetime,
                        help: self
                            .r
                            .tcx
                            .sess
                            .is_nightly_build()
                            .then_some(errors::ParamInNonTrivialAnonConstHelp),
                    })
                    .emit();
            }
        }
    }

    pub(crate) fn report_missing_lifetime_specifiers(
        &mut self,
        lifetime_refs: Vec<MissingLifetime>,
        function_param_lifetimes: Option<(Vec<MissingLifetime>, Vec<ElisionFnParameter>)>,
    ) -> ErrorGuaranteed {
        let num_lifetimes: usize = lifetime_refs.iter().map(|lt| lt.count).sum();
        let spans: Vec<_> = lifetime_refs.iter().map(|lt| lt.span).collect();

        let mut err = struct_span_code_err!(
            self.r.dcx(),
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
        err: &mut Diag<'_>,
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
            .take_while(|rib| {
                !matches!(rib.kind, LifetimeRibKind::Item | LifetimeRibKind::ConstParamTy)
            })
            .flat_map(|rib| rib.bindings.iter())
            .map(|(&ident, &res)| (ident, res))
            .filter(|(ident, _)| ident.name != kw::UnderscoreLifetime)
            .collect();
        debug!(?in_scope_lifetimes);

        let mut maybe_static = false;
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
                    format!("`{ident}`")
                } else {
                    format!("argument {}", index + 1)
                };

                if lifetime_count == 1 {
                    m.push_str(&help_name[..])
                } else {
                    m.push_str(&format!("one of {help_name}'s {lifetime_count} lifetimes")[..])
                }
            }

            if num_params == 0 {
                err.help(
                    "this function's return type contains a borrowed value, but there is no value \
                     for it to be borrowed from",
                );
                if in_scope_lifetimes.is_empty() {
                    maybe_static = true;
                    in_scope_lifetimes = vec![(
                        Ident::with_dummy_span(kw::StaticLifetime),
                        (DUMMY_NODE_ID, LifetimeRes::Static { suppress_elision_warning: false }),
                    )];
                }
            } else if elided_len == 0 {
                err.help(
                    "this function's return type contains a borrowed value with an elided \
                     lifetime, but the lifetime cannot be derived from the arguments",
                );
                if in_scope_lifetimes.is_empty() {
                    maybe_static = true;
                    in_scope_lifetimes = vec![(
                        Ident::with_dummy_span(kw::StaticLifetime),
                        (DUMMY_NODE_ID, LifetimeRes::Static { suppress_elision_warning: false }),
                    )];
                }
            } else if num_params == 1 {
                err.help(format!(
                    "this function's return type contains a borrowed value, but the signature does \
                     not say which {m} it is borrowed from",
                ));
            } else {
                err.help(format!(
                    "this function's return type contains a borrowed value, but the signature does \
                     not say whether it is borrowed from {m}",
                ));
            }
        }

        #[allow(rustc::symbol_intern_string_literal)]
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
                (lt.span.shrink_to_hi(), format!("{existing_name} "))
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
                    |err, higher_ranked, span, message, intro_sugg, _| {
                        err.multipart_suggestion_verbose(
                            message,
                            std::iter::once((span, intro_sugg))
                                .chain(spans_suggs.clone())
                                .collect(),
                            Applicability::MaybeIncorrect,
                        );
                        higher_ranked
                    },
                );
            }
            1 => {
                let post = if maybe_static {
                    let owned = if let [lt] = &lifetime_refs[..]
                        && lt.kind != MissingLifetimeKind::Ampersand
                    {
                        ", or if you will only have owned values"
                    } else {
                        ""
                    };
                    format!(
                        ", but this is uncommon unless you're returning a borrowed value from a \
                         `const` or a `static`{owned}",
                    )
                } else {
                    String::new()
                };
                err.multipart_suggestion_verbose(
                    format!("consider using the `{existing_name}` lifetime{post}"),
                    spans_suggs,
                    Applicability::MaybeIncorrect,
                );
                if maybe_static {
                    // FIXME: what follows are general suggestions, but we'd want to perform some
                    // minimal flow analysis to provide more accurate suggestions. For example, if
                    // we identified that the return expression references only one argument, we
                    // would suggest borrowing only that argument, and we'd skip the prior
                    // "use `'static`" suggestion entirely.
                    if let [lt] = &lifetime_refs[..]
                        && (lt.kind == MissingLifetimeKind::Ampersand
                            || lt.kind == MissingLifetimeKind::Underscore)
                    {
                        let pre = if lt.kind == MissingLifetimeKind::Ampersand
                            && let Some((kind, _span)) = self.diag_metadata.current_function
                            && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = kind
                            && !sig.decl.inputs.is_empty()
                            && let sugg = sig
                                .decl
                                .inputs
                                .iter()
                                .filter_map(|param| {
                                    if param.ty.span.contains(lt.span) {
                                        // We don't want to suggest `fn elision(_: &fn() -> &i32)`
                                        // when we have `fn elision(_: fn() -> &i32)`
                                        None
                                    } else if let TyKind::CVarArgs = param.ty.kind {
                                        // Don't suggest `&...` for ffi fn with varargs
                                        None
                                    } else if let TyKind::ImplTrait(..) = &param.ty.kind {
                                        // We handle these in the next `else if` branch.
                                        None
                                    } else {
                                        Some((param.ty.span.shrink_to_lo(), "&".to_string()))
                                    }
                                })
                                .collect::<Vec<_>>()
                            && !sugg.is_empty()
                        {
                            let (the, s) = if sig.decl.inputs.len() == 1 {
                                ("the", "")
                            } else {
                                ("one of the", "s")
                            };
                            err.multipart_suggestion_verbose(
                                format!(
                                    "instead, you are more likely to want to change {the} \
                                     argument{s} to be borrowed...",
                                ),
                                sugg,
                                Applicability::MaybeIncorrect,
                            );
                            "...or alternatively, you might want"
                        } else if (lt.kind == MissingLifetimeKind::Ampersand
                            || lt.kind == MissingLifetimeKind::Underscore)
                            && let Some((kind, _span)) = self.diag_metadata.current_function
                            && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = kind
                            && let ast::FnRetTy::Ty(ret_ty) = &sig.decl.output
                            && !sig.decl.inputs.is_empty()
                            && let arg_refs = sig
                                .decl
                                .inputs
                                .iter()
                                .filter_map(|param| match &param.ty.kind {
                                    TyKind::ImplTrait(_, bounds) => Some(bounds),
                                    _ => None,
                                })
                                .flat_map(|bounds| bounds.into_iter())
                                .collect::<Vec<_>>()
                            && !arg_refs.is_empty()
                        {
                            // We have a situation like
                            // fn g(mut x: impl Iterator<Item = &()>) -> Option<&()>
                            // So we look at every ref in the trait bound. If there's any, we
                            // suggest
                            // fn g<'a>(mut x: impl Iterator<Item = &'a ()>) -> Option<&'a ()>
                            let mut lt_finder =
                                LifetimeFinder { lifetime: lt.span, found: None, seen: vec![] };
                            for bound in arg_refs {
                                if let ast::GenericBound::Trait(trait_ref) = bound {
                                    lt_finder.visit_trait_ref(&trait_ref.trait_ref);
                                }
                            }
                            lt_finder.visit_ty(ret_ty);
                            let spans_suggs: Vec<_> = lt_finder
                                .seen
                                .iter()
                                .filter_map(|ty| match &ty.kind {
                                    TyKind::Ref(_, mut_ty) => {
                                        let span = ty.span.with_hi(mut_ty.ty.span.lo());
                                        Some((span, "&'a ".to_string()))
                                    }
                                    _ => None,
                                })
                                .collect();
                            self.suggest_introducing_lifetime(
                                err,
                                None,
                                |err, higher_ranked, span, message, intro_sugg, _| {
                                    err.multipart_suggestion_verbose(
                                        message,
                                        std::iter::once((span, intro_sugg))
                                            .chain(spans_suggs.clone())
                                            .collect(),
                                        Applicability::MaybeIncorrect,
                                    );
                                    higher_ranked
                                },
                            );
                            "alternatively, you might want"
                        } else {
                            "instead, you are more likely to want"
                        };
                        let mut owned_sugg = lt.kind == MissingLifetimeKind::Ampersand;
                        let mut sugg = vec![(lt.span, String::new())];
                        if let Some((kind, _span)) = self.diag_metadata.current_function
                            && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = kind
                            && let ast::FnRetTy::Ty(ty) = &sig.decl.output
                        {
                            let mut lt_finder =
                                LifetimeFinder { lifetime: lt.span, found: None, seen: vec![] };
                            lt_finder.visit_ty(&ty);

                            if let [Ty { span, kind: TyKind::Ref(_, mut_ty), .. }] =
                                &lt_finder.seen[..]
                            {
                                // We might have a situation like
                                // fn g(mut x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()>
                                // but `lt.span` only points at `'_`, so to suggest `-> Option<()>`
                                // we need to find a more accurate span to end up with
                                // fn g<'a>(mut x: impl Iterator<Item = &'_ ()>) -> Option<()>
                                sugg = vec![(span.with_hi(mut_ty.ty.span.lo()), String::new())];
                                owned_sugg = true;
                            }
                            if let Some(ty) = lt_finder.found {
                                if let TyKind::Path(None, path) = &ty.kind {
                                    // Check if the path being borrowed is likely to be owned.
                                    let path: Vec<_> = Segment::from_path(path);
                                    match self.resolve_path(&path, Some(TypeNS), None) {
                                        PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                                            match module.res() {
                                                Some(Res::PrimTy(PrimTy::Str)) => {
                                                    // Don't suggest `-> str`, suggest `-> String`.
                                                    sugg = vec![(
                                                        lt.span.with_hi(ty.span.hi()),
                                                        "String".to_string(),
                                                    )];
                                                }
                                                Some(Res::PrimTy(..)) => {}
                                                Some(Res::Def(
                                                    DefKind::Struct
                                                    | DefKind::Union
                                                    | DefKind::Enum
                                                    | DefKind::ForeignTy
                                                    | DefKind::AssocTy
                                                    | DefKind::OpaqueTy
                                                    | DefKind::TyParam,
                                                    _,
                                                )) => {}
                                                _ => {
                                                    // Do not suggest in all other cases.
                                                    owned_sugg = false;
                                                }
                                            }
                                        }
                                        PathResult::NonModule(res) => {
                                            match res.base_res() {
                                                Res::PrimTy(PrimTy::Str) => {
                                                    // Don't suggest `-> str`, suggest `-> String`.
                                                    sugg = vec![(
                                                        lt.span.with_hi(ty.span.hi()),
                                                        "String".to_string(),
                                                    )];
                                                }
                                                Res::PrimTy(..) => {}
                                                Res::Def(
                                                    DefKind::Struct
                                                    | DefKind::Union
                                                    | DefKind::Enum
                                                    | DefKind::ForeignTy
                                                    | DefKind::AssocTy
                                                    | DefKind::OpaqueTy
                                                    | DefKind::TyParam,
                                                    _,
                                                ) => {}
                                                _ => {
                                                    // Do not suggest in all other cases.
                                                    owned_sugg = false;
                                                }
                                            }
                                        }
                                        _ => {
                                            // Do not suggest in all other cases.
                                            owned_sugg = false;
                                        }
                                    }
                                }
                                if let TyKind::Slice(inner_ty) = &ty.kind {
                                    // Don't suggest `-> [T]`, suggest `-> Vec<T>`.
                                    sugg = vec![
                                        (lt.span.with_hi(inner_ty.span.lo()), "Vec<".to_string()),
                                        (ty.span.with_lo(inner_ty.span.hi()), ">".to_string()),
                                    ];
                                }
                            }
                        }
                        if owned_sugg {
                            err.multipart_suggestion_verbose(
                                format!("{pre} to return an owned value"),
                                sugg,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
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

fn mk_where_bound_predicate(
    path: &Path,
    poly_trait_ref: &ast::PolyTraitRef,
    ty: &Ty,
) -> Option<ast::WhereBoundPredicate> {
    let modified_segments = {
        let mut segments = path.segments.clone();
        let [preceding @ .., second_last, last] = segments.as_mut_slice() else {
            return None;
        };
        let mut segments = ThinVec::from(preceding);

        let added_constraint = ast::AngleBracketedArg::Constraint(ast::AssocItemConstraint {
            id: DUMMY_NODE_ID,
            ident: last.ident,
            gen_args: None,
            kind: ast::AssocItemConstraintKind::Equality {
                term: ast::Term::Ty(ast::ptr::P(ast::Ty {
                    kind: ast::TyKind::Path(None, poly_trait_ref.trait_ref.path.clone()),
                    id: DUMMY_NODE_ID,
                    span: DUMMY_SP,
                    tokens: None,
                })),
            },
            span: DUMMY_SP,
        });

        match second_last.args.as_deref_mut() {
            Some(ast::GenericArgs::AngleBracketed(ast::AngleBracketedArgs { args, .. })) => {
                args.push(added_constraint);
            }
            Some(_) => return None,
            None => {
                second_last.args =
                    Some(ast::ptr::P(ast::GenericArgs::AngleBracketed(ast::AngleBracketedArgs {
                        args: ThinVec::from([added_constraint]),
                        span: DUMMY_SP,
                    })));
            }
        }

        segments.push(second_last.clone());
        segments
    };

    let new_where_bound_predicate = ast::WhereBoundPredicate {
        bound_generic_params: ThinVec::new(),
        bounded_ty: ast::ptr::P(ty.clone()),
        bounds: vec![ast::GenericBound::Trait(ast::PolyTraitRef {
            bound_generic_params: ThinVec::new(),
            modifiers: ast::TraitBoundModifiers::NONE,
            trait_ref: ast::TraitRef {
                path: ast::Path { segments: modified_segments, span: DUMMY_SP, tokens: None },
                ref_id: DUMMY_NODE_ID,
            },
            span: DUMMY_SP,
        })],
    };

    Some(new_where_bound_predicate)
}

/// Report lifetime/lifetime shadowing as an error.
pub(super) fn signal_lifetime_shadowing(sess: &Session, orig: Ident, shadower: Ident) {
    struct_span_code_err!(
        sess.dcx(),
        shadower.span,
        E0496,
        "lifetime name `{}` shadows a lifetime name that is already in scope",
        orig.name,
    )
    .with_span_label(orig.span, "first declared here")
    .with_span_label(shadower.span, format!("lifetime `{}` already in scope", orig.name))
    .emit();
}

struct LifetimeFinder<'ast> {
    lifetime: Span,
    found: Option<&'ast Ty>,
    seen: Vec<&'ast Ty>,
}

impl<'ast> Visitor<'ast> for LifetimeFinder<'ast> {
    fn visit_ty(&mut self, t: &'ast Ty) {
        if let TyKind::Ref(_, mut_ty) | TyKind::PinnedRef(_, mut_ty) = &t.kind {
            self.seen.push(t);
            if t.span.lo() == self.lifetime.lo() {
                self.found = Some(&mut_ty.ty);
            }
        }
        walk_ty(self, t)
    }
}

/// Shadowing involving a label is only a warning for historical reasons.
//FIXME: make this a proper lint.
pub(super) fn signal_label_shadowing(sess: &Session, orig: Span, shadower: Ident) {
    let name = shadower.name;
    let shadower = shadower.span;
    sess.dcx()
        .struct_span_warn(
            shadower,
            format!("label name `{name}` shadows a label name that is already in scope"),
        )
        .with_span_label(orig, "first declared here")
        .with_span_label(shadower, format!("label `{name}` already in scope"))
        .emit();
}
