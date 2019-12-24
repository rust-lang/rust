use crate::diagnostics::{ImportSuggestion, TypoSuggestion};
use crate::late::{LateResolutionVisitor, RibKind};
use crate::path_names_to_string;
use crate::{CrateLint, Module, ModuleKind, ModuleOrUniformRoot};
use crate::{PathResult, PathSource, Segment};

use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use log::debug;
use rustc::hir::def::Namespace::{self, *};
use rustc::hir::def::{self, CtorKind, DefKind};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::hir::PrimTy;
use rustc::session::config::nightly_options;
use rustc_data_structures::fx::FxHashSet;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::kw;
use rustc_span::Span;
use syntax::ast::{self, Expr, ExprKind, Ident, NodeId, Path, Ty, TyKind};
use syntax::util::lev_distance::find_best_match_for_name;

use rustc_error_codes::*;

type Res = def::Res<ast::NodeId>;

/// A field or associated item from self type suggested in case of resolution failure.
enum AssocSuggestion {
    Field,
    MethodWithSelf,
    AssocItem,
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
    };
    let enum_path_string = path_names_to_string(&enum_path);

    (variant_path_string, enum_path_string)
}

impl<'a> LateResolutionVisitor<'a, '_> {
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
        let is_enum_variant = &|res| {
            if let Res::Def(DefKind::Variant, _) = res { true } else { false }
        };

        // Make the base error.
        let expected = source.descr_expected();
        let path_str = Segment::names_to_string(path);
        let item_str = path.last().unwrap().ident;
        let code = source.error_code(res.is_some());
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
                    Res::Def(DefKind::Ctor(..), _)
                    | Res::Def(DefKind::Method, _)
                    | Res::Def(DefKind::Const, _)
                    | Res::Def(DefKind::AssocConst, _)
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
                format!("not found in {}", mod_str),
                item_span,
                false,
            )
        };

        let code = DiagnosticId::Error(code.into());
        let mut err = self.r.session.struct_span_err_with_code(base_span, &base_msg, code);

        // Emit help message for fake-self from other languages (e.g., `this` in Javascript).
        if ["this", "my"].contains(&&*item_str.as_str())
            && self.self_value_is_available(path[0].ident.span, span)
        {
            err.span_suggestion(
                span,
                "did you mean",
                "self".to_string(),
                Applicability::MaybeIncorrect,
            );
        }

        // Emit special messages for unresolved `Self` and `self`.
        if is_self_type(path, ns) {
            syntax::diagnostic_used!(E0411);
            err.code(DiagnosticId::Error("E0411".into()));
            err.span_label(
                span,
                format!("`Self` is only available in impls, traits, and type definitions"),
            );
            return (err, Vec::new());
        }
        if is_self_value(path, ns) {
            debug!("smart_resolve_path_fragment: E0424, source={:?}", source);

            syntax::diagnostic_used!(E0424);
            err.code(DiagnosticId::Error("E0424".into()));
            err.span_label(span, match source {
                PathSource::Pat => format!(
                    "`self` value is a keyword and may not be bound to variables or shadowed",
                ),
                _ => format!(
                    "`self` value is a keyword only available in methods with a `self` parameter",
                ),
            });
            if let Some(span) = &self.diagnostic_metadata.current_function {
                err.span_label(*span, "this function doesn't have a `self` parameter");
            }
            return (err, Vec::new());
        }

        // Try to lookup name in more relaxed fashion for better error reporting.
        let ident = path.last().unwrap().ident;
        let candidates = self
            .r
            .lookup_import_candidates(ident, ns, is_expected)
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
            let enum_candidates = self.r.lookup_import_candidates(ident, ns, is_enum_variant);
            let mut enum_candidates = enum_candidates
                .iter()
                .map(|suggestion| import_candidate_to_enum_paths(&suggestion))
                .collect::<Vec<_>>();
            enum_candidates.sort();

            if !enum_candidates.is_empty() {
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
                            "try",
                            format!("self.{}", path_str),
                            Applicability::MachineApplicable,
                        );
                    }
                    AssocSuggestion::MethodWithSelf | AssocSuggestion::AssocItem => {
                        err.span_suggestion(
                            span,
                            "try",
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
        let levenshtein_worked = self.r.add_typo_suggestion(&mut err, typo_sugg, ident_span);

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
                return (err, candidates);
            }
        }

        // Fallback label.
        if !levenshtein_worked {
            err.span_label(base_span, fallback_label);
            self.type_ascription_suggestion(&mut err, base_span);
            match self.diagnostic_metadata.current_let_binding {
                Some((pat_sp, Some(ty_sp), None)) if ty_sp.contains(base_span) && could_be_expr => {
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
        (err, candidates)
    }

    /// Check if the source is call expression and the first argument is `self`. If true,
    /// return the span of whole call and the span for all arguments expect the first one (`self`).
    fn call_has_self_arg(&self, source: PathSource<'_>) -> Option<(Span, Option<Span>)> {
        let mut has_self_arg = None;
        if let PathSource::Expr(parent) = source {
            match &parent?.kind {
                ExprKind::Call(_, args) if args.len() > 0 => {
                    let mut expr_kind = &args[0].kind;
                    loop {
                        match expr_kind {
                            ExprKind::Path(_, arg_name) if arg_name.segments.len() == 1 => {
                                if arg_name.segments[0].ident.name == kw::SelfLower {
                                    let call_span = parent.unwrap().span;
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
        return has_self_arg;
    }

    fn followed_by_brace(&self, span: Span) -> (bool, Option<(Span, String)>) {
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
        let followed_by_brace = match sm.span_to_snippet(sp) {
            Ok(ref snippet) if snippet == "{" => true,
            _ => false,
        };
        // In case this could be a struct literal that needs to be surrounded
        // by parenthesis, find the appropriate span.
        let mut i = 0;
        let mut closing_brace = None;
        loop {
            sp = sm.next_point(sp);
            match sm.span_to_snippet(sp) {
                Ok(ref snippet) => {
                    if snippet == "}" {
                        let sp = span.to(sp);
                        if let Ok(snippet) = sm.span_to_snippet(sp) {
                            closing_brace = Some((sp, snippet));
                        }
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
        return (followed_by_brace, closing_brace);
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
            let mut suggested = false;
            match source {
                PathSource::Expr(Some(parent)) => {
                    suggested = path_sep(err, &parent);
                }
                PathSource::Expr(None) if followed_by_brace == true => {
                    if let Some((sp, snippet)) = closing_brace {
                        err.span_suggestion(
                            sp,
                            "surround the struct literal with parenthesis",
                            format!("({})", snippet),
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_label(
                            span, // Note the parenthesis surrounding the suggestion below
                            format!("did you mean `({} {{ /* fields */ }})`?", path_str),
                        );
                    }
                    suggested = true;
                }
                _ => {}
            }
            if !suggested {
                if let Some(span) = self.r.definitions.opt_span(def_id) {
                    err.span_label(span, &format!("`{}` defined here", path_str));
                }
                err.span_label(span, format!("did you mean `{} {{ /* fields */ }}`?", path_str));
            }
        };

        match (res, source) {
            (Res::Def(DefKind::Macro(MacroKind::Bang), _), _) => {
                err.span_suggestion(
                    span,
                    "use `!` to invoke the macro",
                    format!("{}!", path_str),
                    Applicability::MaybeIncorrect,
                );
                if path_str == "try" && span.rust_2015() {
                    err.note("if you want the `try` keyword, you need to be in the 2018 edition");
                }
            }
            (Res::Def(DefKind::TyAlias, _), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if nightly_options::is_nightly_build() {
                    err.note("did you mean to use a trait alias?");
                }
            }
            (Res::Def(DefKind::Mod, _), PathSource::Expr(Some(parent))) => {
                if !path_sep(err, &parent) {
                    return false;
                }
            }
            (Res::Def(DefKind::Enum, def_id), PathSource::TupleStruct)
            | (Res::Def(DefKind::Enum, def_id), PathSource::Expr(..)) => {
                if let Some(variants) = self.collect_enum_variants(def_id) {
                    if !variants.is_empty() {
                        let msg = if variants.len() == 1 {
                            "try using the enum's variant"
                        } else {
                            "try using one of the enum's variants"
                        };

                        err.span_suggestions(
                            span,
                            msg,
                            variants.iter().map(path_names_to_string),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else {
                    err.note("did you mean to use one of the enum's variants?");
                }
            }
            (Res::Def(DefKind::Struct, def_id), _) if ns == ValueNS => {
                if let Some((ctor_def, ctor_vis)) = self.r.struct_constructors.get(&def_id).cloned()
                {
                    let accessible_ctor =
                        self.r.is_accessible_from(ctor_vis, self.parent_scope.module);
                    if is_expected(ctor_def) && !accessible_ctor {
                        err.span_label(
                            span,
                            format!("constructor is not visible here due to private fields"),
                        );
                    }
                } else {
                    bad_struct_syntax_suggestion(def_id);
                }
            }
            (Res::Def(DefKind::Union, def_id), _)
            | (Res::Def(DefKind::Variant, def_id), _)
            | (Res::Def(DefKind::Ctor(_, CtorKind::Fictive), def_id), _)
                if ns == ValueNS =>
            {
                bad_struct_syntax_suggestion(def_id);
            }
            (Res::Def(DefKind::Ctor(_, CtorKind::Fn), def_id), _) if ns == ValueNS => {
                if let Some(span) = self.r.definitions.opt_span(def_id) {
                    err.span_label(span, &format!("`{}` defined here", path_str));
                }
                err.span_label(span, format!("did you mean `{}( /* fields */ )`?", path_str));
            }
            (Res::SelfTy(..), _) if ns == ValueNS => {
                err.span_label(span, fallback_label);
                err.note("can't use `Self` as a constructor, you must use the implemented struct");
            }
            (Res::Def(DefKind::TyAlias, _), _) | (Res::Def(DefKind::AssocTy, _), _)
                if ns == ValueNS =>
            {
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
                        Res::Def(DefKind::Struct, did) | Res::Def(DefKind::Union, did)
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

        for assoc_type_ident in &self.diagnostic_metadata.current_trait_assoc_types {
            if *assoc_type_ident == ident {
                return Some(AssocSuggestion::AssocItem);
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
                    return Some(if self.r.has_self.contains(&res.def_id()) {
                        AssocSuggestion::MethodWithSelf
                    } else {
                        AssocSuggestion::AssocItem
                    });
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
                                self.r
                                    .crate_loader
                                    .maybe_process_path_extern(ident.name, ident.span)
                                    .and_then(|crate_id| {
                                        let crate_mod = Res::Def(
                                            DefKind::Mod,
                                            DefId { krate: crate_id, index: CRATE_DEF_INDEX },
                                        );

                                        if filter_fn(crate_mod) {
                                            Some(TypoSuggestion::from_res(ident.name, crate_mod))
                                        } else {
                                            None
                                        }
                                    })
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
            names.iter().map(|suggestion| &suggestion.candidate),
            &name.as_str(),
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
        let cm = self.r.session.source_map();
        start.to(cm.next_point(start))
    }

    fn type_ascription_suggestion(&self, err: &mut DiagnosticBuilder<'_>, base_span: Span) {
        let cm = self.r.session.source_map();
        let base_snippet = cm.span_to_snippet(base_span);
        if let Some(sp) = self.diagnostic_metadata.current_type_ascription.last() {
            let mut sp = *sp;
            loop {
                // Try to find the `:`; bail on first non-':' / non-whitespace.
                sp = cm.next_point(sp);
                if let Ok(snippet) = cm.span_to_snippet(sp.to(cm.next_point(sp))) {
                    let line_sp = cm.lookup_char_pos(sp.hi()).line;
                    let line_base_sp = cm.lookup_char_pos(base_span.lo()).line;
                    if snippet == ":" {
                        let mut show_label = true;
                        if line_sp != line_base_sp {
                            err.span_suggestion_short(
                                sp,
                                "did you mean to use `;` here instead?",
                                ";".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            let colon_sp = self.get_colon_suggestion_span(sp);
                            let after_colon_sp =
                                self.get_colon_suggestion_span(colon_sp.shrink_to_hi());
                            if !cm
                                .span_to_snippet(after_colon_sp)
                                .map(|s| s == " ")
                                .unwrap_or(false)
                            {
                                err.span_suggestion(
                                    colon_sp,
                                    "maybe you meant to write a path separator here",
                                    "::".to_string(),
                                    Applicability::MaybeIncorrect,
                                );
                                show_label = false;
                            }
                            if let Ok(base_snippet) = base_snippet {
                                let mut sp = after_colon_sp;
                                for _ in 0..100 {
                                    // Try to find an assignment
                                    sp = cm.next_point(sp);
                                    let snippet = cm.span_to_snippet(sp.to(cm.next_point(sp)));
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
                        break;
                    } else if !snippet.trim().is_empty() {
                        debug!("tried to find type ascription `:` token, couldn't find it");
                        break;
                    }
                } else {
                    break;
                }
            }
        }
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
                        let path = Path { span: name_binding.span, segments: path_segments };
                        result = Some((module, ImportSuggestion { did: Some(def_id), path }));
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

    fn collect_enum_variants(&mut self, def_id: DefId) -> Option<Vec<Path>> {
        self.find_module(def_id).map(|(enum_module, enum_import_suggestion)| {
            let mut variants = Vec::new();
            enum_module.for_each_child(self.r, |_, ident, _, name_binding| {
                if let Res::Def(DefKind::Variant, _) = name_binding.res() {
                    let mut segms = enum_import_suggestion.path.segments.clone();
                    segms.push(ast::PathSegment::from_ident(ident));
                    variants.push(Path { span: name_binding.span, segments: segms });
                }
            });
            variants
        })
    }
}
