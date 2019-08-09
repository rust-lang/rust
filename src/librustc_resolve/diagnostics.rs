use std::cmp::Reverse;

use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use log::debug;
use rustc::hir::def::{self, DefKind, CtorKind, NonMacroAttrKind};
use rustc::hir::def::Namespace::{self, *};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::PrimTy;
use rustc::session::{Session, config::nightly_options};
use rustc::ty::{self, DefIdTree};
use rustc::util::nodemap::FxHashSet;
use syntax::ast::{self, Expr, ExprKind, Ident, NodeId, Path, Ty, TyKind};
use syntax::ext::base::MacroKind;
use syntax::feature_gate::BUILTIN_ATTRIBUTES;
use syntax::symbol::{Symbol, kw};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{BytePos, Span};

use crate::resolve_imports::{ImportDirective, ImportDirectiveSubclass, ImportResolver};
use crate::{is_self_type, is_self_value, path_names_to_string, KNOWN_TOOLS};
use crate::{CrateLint, LegacyScope, Module, ModuleKind, ModuleOrUniformRoot};
use crate::{PathResult, PathSource, ParentScope, Resolver, RibKind, Scope, ScopeSet, Segment};

type Res = def::Res<ast::NodeId>;

/// A vector of spans and replacements, a message and applicability.
crate type Suggestion = (Vec<(Span, String)>, String, Applicability);

/// A field or associated item from self type suggested in case of resolution failure.
enum AssocSuggestion {
    Field,
    MethodWithSelf,
    AssocItem,
}

struct TypoSuggestion {
    candidate: Symbol,
    res: Res,
}

impl TypoSuggestion {
    fn from_res(candidate: Symbol, res: Res) -> TypoSuggestion {
        TypoSuggestion { candidate, res }
    }
}

/// A free importable items suggested in case of resolution failure.
crate struct ImportSuggestion {
    did: Option<DefId>,
    pub path: Path,
}

fn add_typo_suggestion(
    err: &mut DiagnosticBuilder<'_>, suggestion: Option<TypoSuggestion>, span: Span
) -> bool {
    if let Some(suggestion) = suggestion {
        let msg = format!(
            "{} {} with a similar name exists", suggestion.res.article(), suggestion.res.descr()
        );
        err.span_suggestion(
            span, &msg, suggestion.candidate.to_string(), Applicability::MaybeIncorrect
        );
        return true;
    }
    false
}

fn add_module_candidates(
    module: Module<'_>, names: &mut Vec<TypoSuggestion>, filter_fn: &impl Fn(Res) -> bool
) {
    for (&(ident, _), resolution) in module.resolutions.borrow().iter() {
        if let Some(binding) = resolution.borrow().binding {
            let res = binding.res();
            if filter_fn(res) {
                names.push(TypoSuggestion::from_res(ident.name, res));
            }
        }
    }
}

impl<'a> Resolver<'a> {
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
        let (base_msg, fallback_label, base_span) = if let Some(res) = res {
            (format!("expected {}, found {} `{}`", expected, res.descr(), path_str),
                format!("not a {}", expected),
                span)
        } else {
            let item_span = path.last().unwrap().ident.span;
            let (mod_prefix, mod_str) = if path.len() == 1 {
                (String::new(), "this scope".to_string())
            } else if path.len() == 2 && path[0].ident.name == kw::PathRoot {
                (String::new(), "the crate root".to_string())
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_prefix = match self.resolve_path_without_parent_scope(
                    mod_path, Some(TypeNS), false, span, CrateLint::No
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) =>
                        module.def_kind(),
                    _ => None,
                }.map_or(String::new(), |kind| format!("{} ", kind.descr()));
                (mod_prefix, format!("`{}`", Segment::names_to_string(mod_path)))
            };
            (format!("cannot find {} `{}` in {}{}", expected, item_str, mod_prefix, mod_str),
                format!("not found in {}", mod_str),
                item_span)
        };

        let code = DiagnosticId::Error(code.into());
        let mut err = self.session.struct_span_err_with_code(base_span, &base_msg, code);

        // Emit help message for fake-self from other languages (e.g., `this` in Javascript).
        if ["this", "my"].contains(&&*item_str.as_str())
            && self.self_value_is_available(path[0].ident.span, span) {
            err.span_suggestion(
                span,
                "did you mean",
                "self".to_string(),
                Applicability::MaybeIncorrect,
            );
        }

        // Emit special messages for unresolved `Self` and `self`.
        if is_self_type(path, ns) {
            __diagnostic_used!(E0411);
            err.code(DiagnosticId::Error("E0411".into()));
            err.span_label(span, format!("`Self` is only available in impls, traits, \
                                          and type definitions"));
            return (err, Vec::new());
        }
        if is_self_value(path, ns) {
            debug!("smart_resolve_path_fragment: E0424, source={:?}", source);

            __diagnostic_used!(E0424);
            err.code(DiagnosticId::Error("E0424".into()));
            err.span_label(span, match source {
                PathSource::Pat => {
                    format!("`self` value is a keyword \
                             and may not be bound to \
                             variables or shadowed")
                }
                _ => {
                    format!("`self` value is a keyword \
                             only available in methods \
                             with `self` parameter")
                }
            });
            return (err, Vec::new());
        }

        // Try to lookup name in more relaxed fashion for better error reporting.
        let ident = path.last().unwrap().ident;
        let candidates = self.lookup_import_candidates(ident, ns, is_expected)
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
                self.lookup_import_candidates(ident, ns, is_enum_variant);
            let mut enum_candidates = enum_candidates.iter()
                .map(|suggestion| {
                    import_candidate_to_enum_paths(&suggestion)
                }).collect::<Vec<_>>();
            enum_candidates.sort();

            if !enum_candidates.is_empty() {
                // Contextualize for E0412 "cannot find type", but don't belabor the point
                // (that it's a variant) for E0573 "expected type, found variant".
                let preamble = if res.is_none() {
                    let others = match enum_candidates.len() {
                        1 => String::new(),
                        2 => " and 1 other".to_owned(),
                        n => format!(" and {} others", n)
                    };
                    format!("there is an enum variant `{}`{}; ",
                            enum_candidates[0].0, others)
                } else {
                    String::new()
                };
                let msg = format!("{}try using the variant's enum", preamble);

                err.span_suggestions(
                    span,
                    &msg,
                    enum_candidates.into_iter()
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
                            err.span_label(
                                span,
                                "a field by this name exists in `Self`",
                            );
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
        }

        // Try Levenshtein algorithm.
        let levenshtein_worked = add_typo_suggestion(
            &mut err, self.lookup_typo_candidate(path, ns, is_expected, span), ident_span
        );

        // Try context-dependent help if relaxed lookup didn't work.
        if let Some(res) = res {
            if self.smart_resolve_context_dependent_help(&mut err,
                                                         span,
                                                         source,
                                                         res,
                                                         &path_str,
                                                         &fallback_label) {
                return (err, candidates);
            }
        }

        // Fallback label.
        if !levenshtein_worked {
            err.span_label(base_span, fallback_label);
            self.type_ascription_suggestion(&mut err, base_span);
        }
        (err, candidates)
    }

    fn followed_by_brace(&self, span: Span) -> (bool, Option<(Span, String)>) {
        // HACK(estebank): find a better way to figure out that this was a
        // parser issue where a struct literal is being used on an expression
        // where a brace being opened means a block is being started. Look
        // ahead for the next text to see if `span` is followed by a `{`.
        let sm = self.session.source_map();
        let mut sp = span;
        loop {
            sp = sm.next_point(sp);
            match sm.span_to_snippet(sp) {
                Ok(ref snippet) => {
                    if snippet.chars().any(|c| { !c.is_whitespace() }) {
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
        return (followed_by_brace, closing_brace)
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

        let path_sep = |err: &mut DiagnosticBuilder<'_>, expr: &Expr| match expr.node {
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

        let mut bad_struct_syntax_suggestion = || {
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
                            span,  // Note the parenthesis surrounding the suggestion below
                            format!("did you mean `({} {{ /* fields */ }})`?", path_str),
                        );
                    }
                    suggested = true;
                },
                _ => {}
            }
            if !suggested {
                err.span_label(
                    span,
                    format!("did you mean `{} {{ /* fields */ }}`?", path_str),
                );
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
                | (Res::Def(DefKind::Enum, def_id), PathSource::Expr(..))  => {
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
            },
            (Res::Def(DefKind::Struct, def_id), _) if ns == ValueNS => {
                if let Some((ctor_def, ctor_vis))
                        = self.struct_constructors.get(&def_id).cloned() {
                    let accessible_ctor = self.is_accessible(ctor_vis);
                    if is_expected(ctor_def) && !accessible_ctor {
                        err.span_label(
                            span,
                            format!("constructor is not visible here due to private fields"),
                        );
                    }
                } else {
                    bad_struct_syntax_suggestion();
                }
            }
            (Res::Def(DefKind::Union, _), _) |
            (Res::Def(DefKind::Variant, _), _) |
            (Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _), _) if ns == ValueNS => {
                bad_struct_syntax_suggestion();
            }
            (Res::SelfTy(..), _) if ns == ValueNS => {
                err.span_label(span, fallback_label);
                err.note("can't use `Self` as a constructor, you must use the implemented struct");
            }
            (Res::Def(DefKind::TyAlias, _), _)
            | (Res::Def(DefKind::AssocTy, _), _) if ns == ValueNS => {
                err.note("can't use a type alias as a constructor");
            }
            _ => return false,
        }
        true
    }

    fn lookup_assoc_candidate<FilterFn>(&mut self,
                                        ident: Ident,
                                        ns: Namespace,
                                        filter_fn: FilterFn)
                                        -> Option<AssocSuggestion>
        where FilterFn: Fn(Res) -> bool
    {
        fn extract_node_id(t: &Ty) -> Option<NodeId> {
            match t.node {
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
            if let Some(node_id) = self.current_self_type.as_ref().and_then(extract_node_id) {
                // Look for a field with the same name in the current self_type.
                if let Some(resolution) = self.partial_res_map.get(&node_id) {
                    match resolution.base_res() {
                        Res::Def(DefKind::Struct, did) | Res::Def(DefKind::Union, did)
                                if resolution.unresolved_segments() == 0 => {
                            if let Some(field_names) = self.field_names.get(&did) {
                                if field_names.iter().any(|&field_name| ident.name == field_name) {
                                    return Some(AssocSuggestion::Field);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        for assoc_type_ident in &self.current_trait_assoc_types {
            if *assoc_type_ident == ident {
                return Some(AssocSuggestion::AssocItem);
            }
        }

        // Look for associated items in the current trait.
        if let Some((module, _)) = self.current_trait_ref {
            if let Ok(binding) = self.resolve_ident_in_module(
                    ModuleOrUniformRoot::Module(module),
                    ident,
                    ns,
                    None,
                    false,
                    module.span,
                ) {
                let res = binding.res();
                if filter_fn(res) {
                    return Some(if self.has_self.contains(&res.def_id()) {
                        AssocSuggestion::MethodWithSelf
                    } else {
                        AssocSuggestion::AssocItem
                    });
                }
            }
        }

        None
    }

    /// Lookup typo candidate in scope for a macro or import.
    fn early_lookup_typo_candidate(
        &mut self,
        scope_set: ScopeSet,
        parent_scope: &ParentScope<'a>,
        ident: Ident,
        filter_fn: &impl Fn(Res) -> bool,
    ) -> Option<TypoSuggestion> {
        let mut suggestions = Vec::new();
        self.visit_scopes(scope_set, parent_scope, ident, |this, scope, use_prelude, _| {
            match scope {
                Scope::DeriveHelpers => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);
                    if filter_fn(res) {
                        for derive in &parent_scope.derives {
                            let parent_scope = ParentScope { derives: Vec::new(), ..*parent_scope };
                            if let Ok((Some(ext), _)) = this.resolve_macro_path(
                                derive, Some(MacroKind::Derive), &parent_scope, false, false
                            ) {
                                suggestions.extend(ext.helper_attrs.iter().map(|name| {
                                    TypoSuggestion::from_res(*name, res)
                                }));
                            }
                        }
                    }
                }
                Scope::MacroRules(legacy_scope) => {
                    if let LegacyScope::Binding(legacy_binding) = legacy_scope {
                        let res = legacy_binding.binding.res();
                        if filter_fn(res) {
                            suggestions.push(
                                TypoSuggestion::from_res(legacy_binding.ident.name, res)
                            )
                        }
                    }
                }
                Scope::CrateRoot => {
                    let root_ident = Ident::new(kw::PathRoot, ident.span);
                    let root_module = this.resolve_crate_root(root_ident);
                    add_module_candidates(root_module, &mut suggestions, filter_fn);
                }
                Scope::Module(module) => {
                    add_module_candidates(module, &mut suggestions, filter_fn);
                }
                Scope::MacroUsePrelude => {
                    suggestions.extend(this.macro_use_prelude.iter().filter_map(|(name, binding)| {
                        let res = binding.res();
                        if filter_fn(res) {
                            Some(TypoSuggestion::from_res(*name, res))
                        } else {
                            None
                        }
                    }));
                }
                Scope::BuiltinAttrs => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Builtin);
                    if filter_fn(res) {
                        suggestions.extend(BUILTIN_ATTRIBUTES.iter().map(|(name, ..)| {
                            TypoSuggestion::from_res(*name, res)
                        }));
                    }
                }
                Scope::LegacyPluginHelpers => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::LegacyPluginHelper);
                    if filter_fn(res) {
                        let plugin_attributes = this.session.plugin_attributes.borrow();
                        suggestions.extend(plugin_attributes.iter().map(|(name, _)| {
                            TypoSuggestion::from_res(*name, res)
                        }));
                    }
                }
                Scope::ExternPrelude => {
                    suggestions.extend(this.extern_prelude.iter().filter_map(|(ident, _)| {
                        let res = Res::Def(DefKind::Mod, DefId::local(CRATE_DEF_INDEX));
                        if filter_fn(res) {
                            Some(TypoSuggestion::from_res(ident.name, res))
                        } else {
                            None
                        }
                    }));
                }
                Scope::ToolPrelude => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                    suggestions.extend(KNOWN_TOOLS.iter().map(|name| {
                        TypoSuggestion::from_res(*name, res)
                    }));
                }
                Scope::StdLibPrelude => {
                    if let Some(prelude) = this.prelude {
                        let mut tmp_suggestions = Vec::new();
                        add_module_candidates(prelude, &mut tmp_suggestions, filter_fn);
                        suggestions.extend(tmp_suggestions.into_iter().filter(|s| {
                            use_prelude || this.is_builtin_macro(s.res.opt_def_id())
                        }));
                    }
                }
                Scope::BuiltinTypes => {
                    let primitive_types = &this.primitive_type_table.primitive_types;
                    suggestions.extend(
                        primitive_types.iter().flat_map(|(name, prim_ty)| {
                            let res = Res::PrimTy(*prim_ty);
                            if filter_fn(res) {
                                Some(TypoSuggestion::from_res(*name, res))
                            } else {
                                None
                            }
                        })
                    )
                }
            }

            None::<()>
        });

        // Make sure error reporting is deterministic.
        suggestions.sort_by_cached_key(|suggestion| suggestion.candidate.as_str());

        match find_best_match_for_name(
            suggestions.iter().map(|suggestion| &suggestion.candidate),
            &ident.as_str(),
            None,
        ) {
            Some(found) if found != ident.name => suggestions
                .into_iter()
                .find(|suggestion| suggestion.candidate == found),
            _ => None,
        }
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
                    add_module_candidates(module, &mut names, &filter_fn);

                    if let ModuleKind::Block(..) = module.kind {
                        // We can see through blocks
                    } else {
                        // Items from the prelude
                        if !module.no_implicit_prelude {
                            names.extend(self.extern_prelude.clone().iter().flat_map(|(ident, _)| {
                                self.crate_loader
                                    .maybe_process_path_extern(ident.name, ident.span)
                                    .and_then(|crate_id| {
                                        let crate_mod = Res::Def(
                                            DefKind::Mod,
                                            DefId {
                                                krate: crate_id,
                                                index: CRATE_DEF_INDEX,
                                            },
                                        );

                                        if filter_fn(crate_mod) {
                                            Some(TypoSuggestion::from_res(ident.name, crate_mod))
                                        } else {
                                            None
                                        }
                                    })
                            }));

                            if let Some(prelude) = self.prelude {
                                add_module_candidates(prelude, &mut names, &filter_fn);
                            }
                        }
                        break;
                    }
                }
            }
            // Add primitive types to the mix
            if filter_fn(Res::PrimTy(PrimTy::Bool)) {
                names.extend(
                    self.primitive_type_table.primitive_types.iter().map(|(name, prim_ty)| {
                        TypoSuggestion::from_res(*name, Res::PrimTy(*prim_ty))
                    })
                )
            }
        } else {
            // Search in module.
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(module) = self.resolve_path_without_parent_scope(
                mod_path, Some(TypeNS), false, span, CrateLint::No
            ) {
                if let ModuleOrUniformRoot::Module(module) = module {
                    add_module_candidates(module, &mut names, &filter_fn);
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
            Some(found) if found != name => names
                .into_iter()
                .find(|suggestion| suggestion.candidate == found),
            _ => None,
        }
    }

    fn lookup_import_candidates_from_module<FilterFn>(&mut self,
                                          lookup_ident: Ident,
                                          namespace: Namespace,
                                          start_module: Module<'a>,
                                          crate_name: Ident,
                                          filter_fn: FilterFn)
                                          -> Vec<ImportSuggestion>
        where FilterFn: Fn(Res) -> bool
    {
        let mut candidates = Vec::new();
        let mut seen_modules = FxHashSet::default();
        let not_local_module = crate_name.name != kw::Crate;
        let mut worklist = vec![(start_module, Vec::<ast::PathSegment>::new(), not_local_module)];

        while let Some((in_module,
                        path_segments,
                        in_module_is_extern)) = worklist.pop() {
            self.populate_module_if_necessary(in_module);

            // We have to visit module children in deterministic order to avoid
            // instabilities in reported imports (#43552).
            in_module.for_each_child_stable(|ident, ns, name_binding| {
                // avoid imports entirely
                if name_binding.is_import() && !name_binding.is_extern_crate() { return; }
                // avoid non-importable candidates as well
                if !name_binding.is_importable() { return; }

                // collect results based on the filter function
                if ident.name == lookup_ident.name && ns == namespace {
                    let res = name_binding.res();
                    if filter_fn(res) {
                        // create the path
                        let mut segms = path_segments.clone();
                        if lookup_ident.span.rust_2018() {
                            // crate-local absolute paths start with `crate::` in edition 2018
                            // FIXME: may also be stabilized for Rust 2015 (Issues #45477, #44660)
                            segms.insert(
                                0, ast::PathSegment::from_ident(crate_name)
                            );
                        }

                        segms.push(ast::PathSegment::from_ident(ident));
                        let path = Path {
                            span: name_binding.span,
                            segments: segms,
                        };
                        // the entity is accessible in the following cases:
                        // 1. if it's defined in the same crate, it's always
                        // accessible (since private entities can be made public)
                        // 2. if it's defined in another crate, it's accessible
                        // only if both the module is public and the entity is
                        // declared as public (due to pruning, we don't explore
                        // outside crate private modules => no need to check this)
                        if !in_module_is_extern || name_binding.vis == ty::Visibility::Public {
                            let did = match res {
                                Res::Def(DefKind::Ctor(..), did) => self.parent(did),
                                _ => res.opt_def_id(),
                            };
                            candidates.push(ImportSuggestion { did, path });
                        }
                    }
                }

                // collect submodules to explore
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));

                    let is_extern_crate_that_also_appears_in_prelude =
                        name_binding.is_extern_crate() &&
                        lookup_ident.span.rust_2018();

                    let is_visible_to_user =
                        !in_module_is_extern || name_binding.vis == ty::Visibility::Public;

                    if !is_extern_crate_that_also_appears_in_prelude && is_visible_to_user {
                        // add the module to the lookup
                        let is_extern = in_module_is_extern || name_binding.is_extern_crate();
                        if seen_modules.insert(module.def_id().unwrap()) {
                            worklist.push((module, path_segments, is_extern));
                        }
                    }
                }
            })
        }

        candidates
    }

    /// When name resolution fails, this method can be used to look up candidate
    /// entities with the expected name. It allows filtering them using the
    /// supplied predicate (which should be used to only accept the types of
    /// definitions expected, e.g., traits). The lookup spans across all crates.
    ///
    /// N.B., the method does not look into imports, but this is not a problem,
    /// since we report the definitions (thus, the de-aliased imports).
    crate fn lookup_import_candidates<FilterFn>(
        &mut self, lookup_ident: Ident, namespace: Namespace, filter_fn: FilterFn
    ) -> Vec<ImportSuggestion>
        where FilterFn: Fn(Res) -> bool
    {
        let mut suggestions = self.lookup_import_candidates_from_module(
            lookup_ident, namespace, self.graph_root, Ident::with_empty_ctxt(kw::Crate), &filter_fn
        );

        if lookup_ident.span.rust_2018() {
            let extern_prelude_names = self.extern_prelude.clone();
            for (ident, _) in extern_prelude_names.into_iter() {
                if let Some(crate_id) = self.crate_loader.maybe_process_path_extern(ident.name,
                                                                                    ident.span) {
                    let crate_root = self.get_module(DefId {
                        krate: crate_id,
                        index: CRATE_DEF_INDEX,
                    });
                    self.populate_module_if_necessary(&crate_root);

                    suggestions.extend(self.lookup_import_candidates_from_module(
                        lookup_ident, namespace, crate_root, ident, &filter_fn));
                }
            }
        }

        suggestions
    }

    fn find_module(&mut self, def_id: DefId) -> Option<(Module<'a>, ImportSuggestion)> {
        let mut result = None;
        let mut seen_modules = FxHashSet::default();
        let mut worklist = vec![(self.graph_root, Vec::new())];

        while let Some((in_module, path_segments)) = worklist.pop() {
            // abort if the module is already found
            if result.is_some() { break; }

            self.populate_module_if_necessary(in_module);

            in_module.for_each_child_stable(|ident, _, name_binding| {
                // abort if the module is already found or if name_binding is private external
                if result.is_some() || !name_binding.vis.is_visible_locally() {
                    return
                }
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));
                    let module_def_id = module.def_id().unwrap();
                    if module_def_id == def_id {
                        let path = Path {
                            span: name_binding.span,
                            segments: path_segments,
                        };
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
            self.populate_module_if_necessary(enum_module);

            let mut variants = Vec::new();
            enum_module.for_each_child_stable(|ident, _, name_binding| {
                if let Res::Def(DefKind::Variant, _) = name_binding.res() {
                    let mut segms = enum_import_suggestion.path.segments.clone();
                    segms.push(ast::PathSegment::from_ident(ident));
                    variants.push(Path {
                        span: name_binding.span,
                        segments: segms,
                    });
                }
            });
            variants
        })
    }

    crate fn unresolved_macro_suggestions(
        &mut self,
        err: &mut DiagnosticBuilder<'a>,
        macro_kind: MacroKind,
        parent_scope: &ParentScope<'a>,
        ident: Ident,
    ) {
        let is_expected = &|res: Res| res.macro_kind() == Some(macro_kind);
        let suggestion = self.early_lookup_typo_candidate(
            ScopeSet::Macro(macro_kind), &parent_scope, ident, is_expected
        );
        add_typo_suggestion(err, suggestion, ident.span);

        if macro_kind == MacroKind::Derive &&
           (ident.as_str() == "Send" || ident.as_str() == "Sync") {
            let msg = format!("unsafe traits like `{}` should be implemented explicitly", ident);
            err.span_note(ident.span, &msg);
        }
        if self.macro_names.contains(&ident.modern()) {
            err.help("have you added the `#[macro_use]` on the module/import?");
        }
    }
}

impl<'a, 'b> ImportResolver<'a, 'b> {
    /// Adds suggestions for a path that cannot be resolved.
    pub(crate) fn make_path_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        debug!("make_path_suggestion: span={:?} path={:?}", span, path);

        match (path.get(0), path.get(1)) {
            // `{{root}}::ident::...` on both editions.
            // On 2015 `{{root}}` is usually added implicitly.
            (Some(fst), Some(snd)) if fst.ident.name == kw::PathRoot &&
                                      !snd.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018.
            (Some(fst), _) if fst.ident.span.rust_2018() &&
                              !fst.ident.is_path_segment_keyword() => {
                // Insert a placeholder that's later replaced by `self`/`super`/etc.
                path.insert(0, Segment::from_ident(Ident::invalid()));
            }
            _ => return None,
        }

        self.make_missing_self_suggestion(span, path.clone(), parent_scope)
            .or_else(|| self.make_missing_crate_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_missing_super_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_external_crate_suggestion(span, path, parent_scope))
    }

    /// Suggest a missing `self::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `self::foo`?
    /// ```
    fn make_missing_self_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `self` and check if that is valid.
        path[0].ident.name = kw::SelfLower;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_self_suggestion: path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, Vec::new()))
        } else {
            None
        }
    }

    /// Suggests a missing `crate::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `crate::foo`?
    /// ```
    fn make_missing_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Crate;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_crate_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                vec![
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>".to_string()
                ],
            ))
        } else {
            None
        }
    }

    /// Suggests a missing `super::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `super::foo`?
    /// ```
    fn make_missing_super_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Super;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_super_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, Vec::new()))
        } else {
            None
        }
    }

    /// Suggests a missing external crate name if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foobar::Baz;
    ///    |     ^^^^^^ did you mean `baz::foobar`?
    /// ```
    ///
    /// Used when importing a submodule of an external crate but missing that crate's
    /// name as the first part of path.
    fn make_external_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        if path[1].ident.span.rust_2015() {
            return None;
        }

        // Sort extern crate names in reverse order to get
        // 1) some consistent ordering for emitted dignostics, and
        // 2) `std` suggestions before `core` suggestions.
        let mut extern_crate_names =
            self.resolver.extern_prelude.iter().map(|(ident, _)| ident.name).collect::<Vec<_>>();
        extern_crate_names.sort_by_key(|name| Reverse(name.as_str()));

        for name in extern_crate_names.into_iter() {
            // Replace first ident with a crate name and check if that is valid.
            path[0].ident.name = name;
            let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
            debug!("make_external_crate_suggestion: name={:?} path={:?} result={:?}",
                    name, path, result);
            if let PathResult::Module(..) = result {
                return Some((path, Vec::new()));
            }
        }

        None
    }

    /// Suggests importing a macro from the root of the crate rather than a module within
    /// the crate.
    ///
    /// ```
    /// help: a macro with this name exists at the root of the crate
    ///    |
    /// LL | use issue_59764::makro;
    ///    |     ^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: this could be because a macro annotated with `#[macro_export]` will be exported
    ///            at the root of the crate instead of the module where it is defined
    /// ```
    pub(crate) fn check_for_module_export_macro(
        &self,
        directive: &'b ImportDirective<'b>,
        module: ModuleOrUniformRoot<'b>,
        ident: Ident,
    ) -> Option<(Option<Suggestion>, Vec<String>)> {
        let mut crate_module = if let ModuleOrUniformRoot::Module(module) = module {
            module
        } else {
            return None;
        };

        while let Some(parent) = crate_module.parent {
            crate_module = parent;
        }

        if ModuleOrUniformRoot::same_def(ModuleOrUniformRoot::Module(crate_module), module) {
            // Don't make a suggestion if the import was already from the root of the
            // crate.
            return None;
        }

        let resolutions = crate_module.resolutions.borrow();
        let resolution = resolutions.get(&(ident, MacroNS))?;
        let binding = resolution.borrow().binding()?;
        if let Res::Def(DefKind::Macro(MacroKind::Bang), _) = binding.res() {
            let module_name = crate_module.kind.name().unwrap();
            let import = match directive.subclass {
                ImportDirectiveSubclass::SingleImport { source, target, .. } if source != target =>
                    format!("{} as {}", source, target),
                _ => format!("{}", ident),
            };

            let mut corrections: Vec<(Span, String)> = Vec::new();
            if !directive.is_nested() {
                // Assume this is the easy case of `use issue_59764::foo::makro;` and just remove
                // intermediate segments.
                corrections.push((directive.span, format!("{}::{}", module_name, import)));
            } else {
                // Find the binding span (and any trailing commas and spaces).
                //   ie. `use a::b::{c, d, e};`
                //                      ^^^
                let (found_closing_brace, binding_span) = find_span_of_binding_until_next_binding(
                    self.resolver.session, directive.span, directive.use_span,
                );
                debug!("check_for_module_export_macro: found_closing_brace={:?} binding_span={:?}",
                       found_closing_brace, binding_span);

                let mut removal_span = binding_span;
                if found_closing_brace {
                    // If the binding span ended with a closing brace, as in the below example:
                    //   ie. `use a::b::{c, d};`
                    //                      ^
                    // Then expand the span of characters to remove to include the previous
                    // binding's trailing comma.
                    //   ie. `use a::b::{c, d};`
                    //                    ^^^
                    if let Some(previous_span) = extend_span_to_previous_binding(
                        self.resolver.session, binding_span,
                    ) {
                        debug!("check_for_module_export_macro: previous_span={:?}", previous_span);
                        removal_span = removal_span.with_lo(previous_span.lo());
                    }
                }
                debug!("check_for_module_export_macro: removal_span={:?}", removal_span);

                // Remove the `removal_span`.
                corrections.push((removal_span, "".to_string()));

                // Find the span after the crate name and if it has nested imports immediatately
                // after the crate name already.
                //   ie. `use a::b::{c, d};`
                //               ^^^^^^^^^
                //   or  `use a::{b, c, d}};`
                //               ^^^^^^^^^^^
                let (has_nested, after_crate_name) = find_span_immediately_after_crate_name(
                    self.resolver.session, module_name, directive.use_span,
                );
                debug!("check_for_module_export_macro: has_nested={:?} after_crate_name={:?}",
                       has_nested, after_crate_name);

                let source_map = self.resolver.session.source_map();

                // Add the import to the start, with a `{` if required.
                let start_point = source_map.start_point(after_crate_name);
                if let Ok(start_snippet) = source_map.span_to_snippet(start_point) {
                    corrections.push((
                        start_point,
                        if has_nested {
                            // In this case, `start_snippet` must equal '{'.
                            format!("{}{}, ", start_snippet, import)
                        } else {
                            // In this case, add a `{`, then the moved import, then whatever
                            // was there before.
                            format!("{{{}, {}", import, start_snippet)
                        }
                    ));
                }

                // Add a `};` to the end if nested, matching the `{` added at the start.
                if !has_nested {
                    corrections.push((source_map.end_point(after_crate_name),
                                     "};".to_string()));
                }
            }

            let suggestion = Some((
                corrections,
                String::from("a macro with this name exists at the root of the crate"),
                Applicability::MaybeIncorrect,
            ));
            let note = vec![
                "this could be because a macro annotated with `#[macro_export]` will be exported \
                 at the root of the crate instead of the module where it is defined".to_string(),
            ];
            Some((suggestion, note))
        } else {
            None
        }
    }
}

/// Given a `binding_span` of a binding within a use statement:
///
/// ```
/// use foo::{a, b, c};
///              ^
/// ```
///
/// then return the span until the next binding or the end of the statement:
///
/// ```
/// use foo::{a, b, c};
///              ^^^
/// ```
pub(crate) fn find_span_of_binding_until_next_binding(
    sess: &Session,
    binding_span: Span,
    use_span: Span,
) -> (bool, Span) {
    let source_map = sess.source_map();

    // Find the span of everything after the binding.
    //   ie. `a, e};` or `a};`
    let binding_until_end = binding_span.with_hi(use_span.hi());

    // Find everything after the binding but not including the binding.
    //   ie. `, e};` or `};`
    let after_binding_until_end = binding_until_end.with_lo(binding_span.hi());

    // Keep characters in the span until we encounter something that isn't a comma or
    // whitespace.
    //   ie. `, ` or ``.
    //
    // Also note whether a closing brace character was encountered. If there
    // was, then later go backwards to remove any trailing commas that are left.
    let mut found_closing_brace = false;
    let after_binding_until_next_binding = source_map.span_take_while(
        after_binding_until_end,
        |&ch| {
            if ch == '}' { found_closing_brace = true; }
            ch == ' ' || ch == ','
        }
    );

    // Combine the two spans.
    //   ie. `a, ` or `a`.
    //
    // Removing these would leave `issue_52891::{d, e};` or `issue_52891::{d, e, };`
    let span = binding_span.with_hi(after_binding_until_next_binding.hi());

    (found_closing_brace, span)
}

/// Given a `binding_span`, return the span through to the comma or opening brace of the previous
/// binding.
///
/// ```
/// use foo::a::{a, b, c};
///               ^^--- binding span
///               |
///               returned span
///
/// use foo::{a, b, c};
///           --- binding span
/// ```
pub(crate) fn extend_span_to_previous_binding(
    sess: &Session,
    binding_span: Span,
) -> Option<Span> {
    let source_map = sess.source_map();

    // `prev_source` will contain all of the source that came before the span.
    // Then split based on a command and take the first (ie. closest to our span)
    // snippet. In the example, this is a space.
    let prev_source = source_map.span_to_prev_source(binding_span).ok()?;

    let prev_comma = prev_source.rsplit(',').collect::<Vec<_>>();
    let prev_starting_brace = prev_source.rsplit('{').collect::<Vec<_>>();
    if prev_comma.len() <= 1 || prev_starting_brace.len() <= 1 {
        return None;
    }

    let prev_comma = prev_comma.first().unwrap();
    let prev_starting_brace = prev_starting_brace.first().unwrap();

    // If the amount of source code before the comma is greater than
    // the amount of source code before the starting brace then we've only
    // got one item in the nested item (eg. `issue_52891::{self}`).
    if prev_comma.len() > prev_starting_brace.len() {
        return None;
    }

    Some(binding_span.with_lo(BytePos(
        // Take away the number of bytes for the characters we've found and an
        // extra for the comma.
        binding_span.lo().0 - (prev_comma.as_bytes().len() as u32) - 1
    )))
}

/// Given a `use_span` of a binding within a use statement, returns the highlighted span and if
/// it is a nested use tree.
///
/// ```
/// use foo::a::{b, c};
///          ^^^^^^^^^^ // false
///
/// use foo::{a, b, c};
///          ^^^^^^^^^^ // true
///
/// use foo::{a, b::{c, d}};
///          ^^^^^^^^^^^^^^^ // true
/// ```
fn find_span_immediately_after_crate_name(
    sess: &Session,
    module_name: Symbol,
    use_span: Span,
) -> (bool, Span) {
    debug!("find_span_immediately_after_crate_name: module_name={:?} use_span={:?}",
           module_name, use_span);
    let source_map = sess.source_map();

    // Using `use issue_59764::foo::{baz, makro};` as an example throughout..
    let mut num_colons = 0;
    // Find second colon.. `use issue_59764:`
    let until_second_colon = source_map.span_take_while(use_span, |c| {
        if *c == ':' { num_colons += 1; }
        match c {
            ':' if num_colons == 2 => false,
            _ => true,
        }
    });
    // Find everything after the second colon.. `foo::{baz, makro};`
    let from_second_colon = use_span.with_lo(until_second_colon.hi() + BytePos(1));

    let mut found_a_non_whitespace_character = false;
    // Find the first non-whitespace character in `from_second_colon`.. `f`
    let after_second_colon = source_map.span_take_while(from_second_colon, |c| {
        if found_a_non_whitespace_character { return false; }
        if !c.is_whitespace() { found_a_non_whitespace_character = true; }
        true
    });

    // Find the first `{` in from_second_colon.. `foo::{`
    let next_left_bracket = source_map.span_through_char(from_second_colon, '{');

    (next_left_bracket == after_second_colon, from_second_colon)
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

/// When an entity with a given name is not available in scope, we search for
/// entities with that name in all crates. This method allows outputting the
/// results of this search in a programmer-friendly way
crate fn show_candidates(
    err: &mut DiagnosticBuilder<'_>,
    // This is `None` if all placement locations are inside expansions
    span: Option<Span>,
    candidates: &[ImportSuggestion],
    better: bool,
    found_use: bool,
) {
    // we want consistent results across executions, but candidates are produced
    // by iterating through a hash map, so make sure they are ordered:
    let mut path_strings: Vec<_> =
        candidates.into_iter().map(|c| path_names_to_string(&c.path)).collect();
    path_strings.sort();

    let better = if better { "better " } else { "" };
    let msg_diff = match path_strings.len() {
        1 => " is found in another module, you can import it",
        _ => "s are found in other modules, you can import them",
    };
    let msg = format!("possible {}candidate{} into scope", better, msg_diff);

    if let Some(span) = span {
        for candidate in &mut path_strings {
            // produce an additional newline to separate the new use statement
            // from the directly following item.
            let additional_newline = if found_use {
                ""
            } else {
                "\n"
            };
            *candidate = format!("use {};\n{}", candidate, additional_newline);
        }

        err.span_suggestions(
            span,
            &msg,
            path_strings.into_iter(),
            Applicability::Unspecified,
        );
    } else {
        let mut msg = msg;
        msg.push(':');
        for candidate in path_strings {
            msg.push('\n');
            msg.push_str(&candidate);
        }
    }
}
