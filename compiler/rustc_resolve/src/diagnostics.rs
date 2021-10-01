use std::cmp::Reverse;
use std::ptr;

use rustc_ast::{self as ast, Path};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_feature::BUILTIN_ATTRIBUTES;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, CtorOf, DefKind, NonMacroAttrKind};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::PrimTy;
use rustc_middle::bug;
use rustc_middle::ty::{self, DefIdTree};
use rustc_session::Session;
use rustc_span::hygiene::MacroKind;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, MultiSpan, Span};
use tracing::debug;

use crate::imports::{Import, ImportKind, ImportResolver};
use crate::path_names_to_string;
use crate::{AmbiguityError, AmbiguityErrorMisc, AmbiguityKind};
use crate::{
    BindingError, CrateLint, HasGenericParams, MacroRulesScope, Module, ModuleOrUniformRoot,
};
use crate::{NameBinding, NameBindingKind, PrivacyError, VisResolutionError};
use crate::{ParentScope, PathResult, ResolutionError, Resolver, Scope, ScopeSet, Segment};

type Res = def::Res<ast::NodeId>;

/// A vector of spans and replacements, a message and applicability.
crate type Suggestion = (Vec<(Span, String)>, String, Applicability);

/// Potential candidate for an undeclared or out-of-scope label - contains the ident of a
/// similarly named label and whether or not it is reachable.
crate type LabelSuggestion = (Ident, bool);

crate enum SuggestionTarget {
    /// The target has a similar name as the name used by the programmer (probably a typo)
    SimilarlyNamed,
    /// The target is the only valid item that can be used in the corresponding context
    SingleItem,
}

crate struct TypoSuggestion {
    pub candidate: Symbol,
    pub res: Res,
    pub target: SuggestionTarget,
}

impl TypoSuggestion {
    crate fn typo_from_res(candidate: Symbol, res: Res) -> TypoSuggestion {
        Self { candidate, res, target: SuggestionTarget::SimilarlyNamed }
    }
    crate fn single_item_from_res(candidate: Symbol, res: Res) -> TypoSuggestion {
        Self { candidate, res, target: SuggestionTarget::SingleItem }
    }
}

/// A free importable items suggested in case of resolution failure.
crate struct ImportSuggestion {
    pub did: Option<DefId>,
    pub descr: &'static str,
    pub path: Path,
    pub accessible: bool,
}

/// Adjust the impl span so that just the `impl` keyword is taken by removing
/// everything after `<` (`"impl<T> Iterator for A<T> {}" -> "impl"`) and
/// everything after the first whitespace (`"impl Iterator for A" -> "impl"`).
///
/// *Attention*: the method used is very fragile since it essentially duplicates the work of the
/// parser. If you need to use this function or something similar, please consider updating the
/// `source_map` functions and this function to something more robust.
fn reduce_impl_span_to_impl_keyword(sm: &SourceMap, impl_span: Span) -> Span {
    let impl_span = sm.span_until_char(impl_span, '<');
    sm.span_until_whitespace(impl_span)
}

impl<'a> Resolver<'a> {
    crate fn add_module_candidates(
        &mut self,
        module: Module<'a>,
        names: &mut Vec<TypoSuggestion>,
        filter_fn: &impl Fn(Res) -> bool,
    ) {
        for (key, resolution) in self.resolutions(module).borrow().iter() {
            if let Some(binding) = resolution.borrow().binding {
                let res = binding.res();
                if filter_fn(res) {
                    names.push(TypoSuggestion::typo_from_res(key.ident.name, res));
                }
            }
        }
    }

    /// Combines an error with provided span and emits it.
    ///
    /// This takes the error provided, combines it with the span and any additional spans inside the
    /// error and emits it.
    crate fn report_error(&self, span: Span, resolution_error: ResolutionError<'_>) {
        self.into_struct_error(span, resolution_error).emit();
    }

    crate fn into_struct_error(
        &self,
        span: Span,
        resolution_error: ResolutionError<'_>,
    ) -> DiagnosticBuilder<'_> {
        match resolution_error {
            ResolutionError::GenericParamsFromOuterFunction(outer_res, has_generic_params) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0401,
                    "can't use generic parameters from outer function",
                );
                err.span_label(span, "use of generic parameter from outer function".to_string());

                let sm = self.session.source_map();
                match outer_res {
                    Res::SelfTy(maybe_trait_defid, maybe_impl_defid) => {
                        if let Some(impl_span) =
                            maybe_impl_defid.and_then(|(def_id, _)| self.opt_span(def_id))
                        {
                            err.span_label(
                                reduce_impl_span_to_impl_keyword(sm, impl_span),
                                "`Self` type implicitly declared here, by this `impl`",
                            );
                        }
                        match (maybe_trait_defid, maybe_impl_defid) {
                            (Some(_), None) => {
                                err.span_label(span, "can't use `Self` here");
                            }
                            (_, Some(_)) => {
                                err.span_label(span, "use a type here instead");
                            }
                            (None, None) => bug!("`impl` without trait nor type?"),
                        }
                        return err;
                    }
                    Res::Def(DefKind::TyParam, def_id) => {
                        if let Some(span) = self.opt_span(def_id) {
                            err.span_label(span, "type parameter from outer function");
                        }
                    }
                    Res::Def(DefKind::ConstParam, def_id) => {
                        if let Some(span) = self.opt_span(def_id) {
                            err.span_label(span, "const parameter from outer function");
                        }
                    }
                    _ => {
                        bug!(
                            "GenericParamsFromOuterFunction should only be used with Res::SelfTy, \
                            DefKind::TyParam or DefKind::ConstParam"
                        );
                    }
                }

                if has_generic_params == HasGenericParams::Yes {
                    // Try to retrieve the span of the function signature and generate a new
                    // message with a local type or const parameter.
                    let sugg_msg = "try using a local generic parameter instead";
                    if let Some((sugg_span, snippet)) = sm.generate_local_type_param_snippet(span) {
                        // Suggest the modification to the user
                        err.span_suggestion(
                            sugg_span,
                            sugg_msg,
                            snippet,
                            Applicability::MachineApplicable,
                        );
                    } else if let Some(sp) = sm.generate_fn_name_span(span) {
                        err.span_label(
                            sp,
                            "try adding a local generic parameter in this method instead"
                                .to_string(),
                        );
                    } else {
                        err.help("try using a local generic parameter instead");
                    }
                }

                err
            }
            ResolutionError::NameAlreadyUsedInParameterList(name, first_use_span) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0403,
                    "the name `{}` is already used for a generic \
                     parameter in this item's generic parameters",
                    name,
                );
                err.span_label(span, "already used");
                err.span_label(first_use_span, format!("first use of `{}`", name));
                err
            }
            ResolutionError::MethodNotMemberOfTrait(method, trait_, candidate) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0407,
                    "method `{}` is not a member of trait `{}`",
                    method,
                    trait_
                );
                err.span_label(span, format!("not a member of trait `{}`", trait_));
                if let Some(candidate) = candidate {
                    err.span_suggestion(
                        method.span,
                        "there is an associated function with a similar name",
                        candidate.to_ident_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
                err
            }
            ResolutionError::TypeNotMemberOfTrait(type_, trait_, candidate) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0437,
                    "type `{}` is not a member of trait `{}`",
                    type_,
                    trait_
                );
                err.span_label(span, format!("not a member of trait `{}`", trait_));
                if let Some(candidate) = candidate {
                    err.span_suggestion(
                        type_.span,
                        "there is an associated type with a similar name",
                        candidate.to_ident_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
                err
            }
            ResolutionError::ConstNotMemberOfTrait(const_, trait_, candidate) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0438,
                    "const `{}` is not a member of trait `{}`",
                    const_,
                    trait_
                );
                err.span_label(span, format!("not a member of trait `{}`", trait_));
                if let Some(candidate) = candidate {
                    err.span_suggestion(
                        const_.span,
                        "there is an associated constant with a similar name",
                        candidate.to_ident_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
                err
            }
            ResolutionError::VariableNotBoundInPattern(binding_error) => {
                let BindingError { name, target, origin, could_be_path } = binding_error;

                let target_sp = target.iter().copied().collect::<Vec<_>>();
                let origin_sp = origin.iter().copied().collect::<Vec<_>>();

                let msp = MultiSpan::from_spans(target_sp.clone());
                let mut err = struct_span_err!(
                    self.session,
                    msp,
                    E0408,
                    "variable `{}` is not bound in all patterns",
                    name,
                );
                for sp in target_sp {
                    err.span_label(sp, format!("pattern doesn't bind `{}`", name));
                }
                for sp in origin_sp {
                    err.span_label(sp, "variable not in all patterns");
                }
                if *could_be_path {
                    let help_msg = format!(
                        "if you meant to match on a variant or a `const` item, consider \
                         making the path in the pattern qualified: `?::{}`",
                        name,
                    );
                    err.span_help(span, &help_msg);
                }
                err
            }
            ResolutionError::VariableBoundWithDifferentMode(variable_name, first_binding_span) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0409,
                    "variable `{}` is bound inconsistently across alternatives separated by `|`",
                    variable_name
                );
                err.span_label(span, "bound in different ways");
                err.span_label(first_binding_span, "first binding");
                err
            }
            ResolutionError::IdentifierBoundMoreThanOnceInParameterList(identifier) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0415,
                    "identifier `{}` is bound more than once in this parameter list",
                    identifier
                );
                err.span_label(span, "used as parameter more than once");
                err
            }
            ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(identifier) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0416,
                    "identifier `{}` is bound more than once in the same pattern",
                    identifier
                );
                err.span_label(span, "used in a pattern more than once");
                err
            }
            ResolutionError::UndeclaredLabel { name, suggestion } => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0426,
                    "use of undeclared label `{}`",
                    name
                );

                err.span_label(span, format!("undeclared label `{}`", name));

                match suggestion {
                    // A reachable label with a similar name exists.
                    Some((ident, true)) => {
                        err.span_label(ident.span, "a label with a similar name is reachable");
                        err.span_suggestion(
                            span,
                            "try using similarly named label",
                            ident.name.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    // An unreachable label with a similar name exists.
                    Some((ident, false)) => {
                        err.span_label(
                            ident.span,
                            "a label with a similar name exists but is unreachable",
                        );
                    }
                    // No similarly-named labels exist.
                    None => (),
                }

                err
            }
            ResolutionError::SelfImportsOnlyAllowedWithin { root, span_with_rename } => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0429,
                    "{}",
                    "`self` imports are only allowed within a { } list"
                );

                // None of the suggestions below would help with a case like `use self`.
                if !root {
                    // use foo::bar::self        -> foo::bar
                    // use foo::bar::self as abc -> foo::bar as abc
                    err.span_suggestion(
                        span,
                        "consider importing the module directly",
                        "".to_string(),
                        Applicability::MachineApplicable,
                    );

                    // use foo::bar::self        -> foo::bar::{self}
                    // use foo::bar::self as abc -> foo::bar::{self as abc}
                    let braces = vec![
                        (span_with_rename.shrink_to_lo(), "{".to_string()),
                        (span_with_rename.shrink_to_hi(), "}".to_string()),
                    ];
                    err.multipart_suggestion(
                        "alternatively, use the multi-path `use` syntax to import `self`",
                        braces,
                        Applicability::MachineApplicable,
                    );
                }
                err
            }
            ResolutionError::SelfImportCanOnlyAppearOnceInTheList => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0430,
                    "`self` import can only appear once in an import list"
                );
                err.span_label(span, "can only appear once in an import list");
                err
            }
            ResolutionError::SelfImportOnlyInImportListWithNonEmptyPrefix => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0431,
                    "`self` import can only appear in an import list with \
                                                a non-empty prefix"
                );
                err.span_label(span, "can only appear in an import list with a non-empty prefix");
                err
            }
            ResolutionError::FailedToResolve { label, suggestion } => {
                let mut err =
                    struct_span_err!(self.session, span, E0433, "failed to resolve: {}", &label);
                err.span_label(span, label);

                if let Some((suggestions, msg, applicability)) = suggestion {
                    err.multipart_suggestion(&msg, suggestions, applicability);
                }

                err
            }
            ResolutionError::CannotCaptureDynamicEnvironmentInFnItem => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0434,
                    "{}",
                    "can't capture dynamic environment in a fn item"
                );
                err.help("use the `|| { ... }` closure form instead");
                err
            }
            ResolutionError::AttemptToUseNonConstantValueInConstant(ident, sugg, current) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0435,
                    "attempt to use a non-constant value in a constant"
                );
                // let foo =...
                //     ^^^ given this Span
                // ------- get this Span to have an applicable suggestion
                let sp =
                    self.session.source_map().span_extend_to_prev_str(ident.span, current, true);
                if sp.lo().0 == 0 {
                    err.span_label(ident.span, &format!("this would need to be a `{}`", sugg));
                } else {
                    let sp = sp.with_lo(BytePos(sp.lo().0 - current.len() as u32));
                    err.span_suggestion(
                        sp,
                        &format!("consider using `{}` instead of `{}`", sugg, current),
                        format!("{} {}", sugg, ident),
                        Applicability::MaybeIncorrect,
                    );
                    err.span_label(span, "non-constant value");
                }
                err
            }
            ResolutionError::BindingShadowsSomethingUnacceptable {
                shadowing_binding_descr,
                name,
                participle,
                article,
                shadowed_binding_descr,
                shadowed_binding_span,
            } => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0530,
                    "{}s cannot shadow {}s",
                    shadowing_binding_descr,
                    shadowed_binding_descr,
                );
                err.span_label(
                    span,
                    format!("cannot be named the same as {} {}", article, shadowed_binding_descr),
                );
                let msg =
                    format!("the {} `{}` is {} here", shadowed_binding_descr, name, participle);
                err.span_label(shadowed_binding_span, msg);
                err
            }
            ResolutionError::ForwardDeclaredGenericParam => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0128,
                    "generic parameters with a default cannot use \
                                                forward declared identifiers"
                );
                err.span_label(
                    span,
                    "defaulted generic parameters cannot be forward declared".to_string(),
                );
                err
            }
            ResolutionError::ParamInTyOfConstParam(name) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0770,
                    "the type of const parameters must not depend on other generic parameters"
                );
                err.span_label(
                    span,
                    format!("the type must not depend on the parameter `{}`", name),
                );
                err
            }
            ResolutionError::ParamInNonTrivialAnonConst { name, is_type } => {
                let mut err = self.session.struct_span_err(
                    span,
                    "generic parameters may not be used in const operations",
                );
                err.span_label(span, &format!("cannot perform const operation using `{}`", name));

                if is_type {
                    err.note("type parameters may not be used in const expressions");
                } else {
                    err.help(&format!(
                        "const parameters may only be used as standalone arguments, i.e. `{}`",
                        name
                    ));
                }

                if self.session.is_nightly_build() {
                    err.help(
                        "use `#![feature(generic_const_exprs)]` to allow generic const expressions",
                    );
                }

                err
            }
            ResolutionError::SelfInGenericParamDefault => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0735,
                    "generic parameters cannot use `Self` in their defaults"
                );
                err.span_label(span, "`Self` in generic parameter default".to_string());
                err
            }
            ResolutionError::UnreachableLabel { name, definition_span, suggestion } => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0767,
                    "use of unreachable label `{}`",
                    name,
                );

                err.span_label(definition_span, "unreachable label defined here");
                err.span_label(span, format!("unreachable label `{}`", name));
                err.note(
                    "labels are unreachable through functions, closures, async blocks and modules",
                );

                match suggestion {
                    // A reachable label with a similar name exists.
                    Some((ident, true)) => {
                        err.span_label(ident.span, "a label with a similar name is reachable");
                        err.span_suggestion(
                            span,
                            "try using similarly named label",
                            ident.name.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    // An unreachable label with a similar name exists.
                    Some((ident, false)) => {
                        err.span_label(
                            ident.span,
                            "a label with a similar name exists but is also unreachable",
                        );
                    }
                    // No similarly-named labels exist.
                    None => (),
                }

                err
            }
        }
    }

    crate fn report_vis_error(&self, vis_resolution_error: VisResolutionError<'_>) {
        match vis_resolution_error {
            VisResolutionError::Relative2018(span, path) => {
                let mut err = self.session.struct_span_err(
                    span,
                    "relative paths are not supported in visibilities on 2018 edition",
                );
                err.span_suggestion(
                    path.span,
                    "try",
                    format!("crate::{}", pprust::path_to_string(&path)),
                    Applicability::MaybeIncorrect,
                );
                err
            }
            VisResolutionError::AncestorOnly(span) => struct_span_err!(
                self.session,
                span,
                E0742,
                "visibilities can only be restricted to ancestor modules"
            ),
            VisResolutionError::FailedToResolve(span, label, suggestion) => {
                self.into_struct_error(span, ResolutionError::FailedToResolve { label, suggestion })
            }
            VisResolutionError::ExpectedFound(span, path_str, res) => {
                let mut err = struct_span_err!(
                    self.session,
                    span,
                    E0577,
                    "expected module, found {} `{}`",
                    res.descr(),
                    path_str
                );
                err.span_label(span, "not a module");
                err
            }
            VisResolutionError::Indeterminate(span) => struct_span_err!(
                self.session,
                span,
                E0578,
                "cannot determine resolution for the visibility"
            ),
            VisResolutionError::ModuleOnly(span) => {
                self.session.struct_span_err(span, "visibility must resolve to a module")
            }
        }
        .emit()
    }

    /// Lookup typo candidate in scope for a macro or import.
    fn early_lookup_typo_candidate(
        &mut self,
        scope_set: ScopeSet<'a>,
        parent_scope: &ParentScope<'a>,
        ident: Ident,
        filter_fn: &impl Fn(Res) -> bool,
    ) -> Option<TypoSuggestion> {
        let mut suggestions = Vec::new();
        let ctxt = ident.span.ctxt();
        self.visit_scopes(scope_set, parent_scope, ctxt, |this, scope, use_prelude, _| {
            match scope {
                Scope::DeriveHelpers(expn_id) => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);
                    if filter_fn(res) {
                        suggestions.extend(
                            this.helper_attrs
                                .get(&expn_id)
                                .into_iter()
                                .flatten()
                                .map(|ident| TypoSuggestion::typo_from_res(ident.name, res)),
                        );
                    }
                }
                Scope::DeriveHelpersCompat => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat);
                    if filter_fn(res) {
                        for derive in parent_scope.derives {
                            let parent_scope = &ParentScope { derives: &[], ..*parent_scope };
                            if let Ok((Some(ext), _)) = this.resolve_macro_path(
                                derive,
                                Some(MacroKind::Derive),
                                parent_scope,
                                false,
                                false,
                            ) {
                                suggestions.extend(
                                    ext.helper_attrs
                                        .iter()
                                        .map(|name| TypoSuggestion::typo_from_res(*name, res)),
                                );
                            }
                        }
                    }
                }
                Scope::MacroRules(macro_rules_scope) => {
                    if let MacroRulesScope::Binding(macro_rules_binding) = macro_rules_scope.get() {
                        let res = macro_rules_binding.binding.res();
                        if filter_fn(res) {
                            suggestions.push(TypoSuggestion::typo_from_res(
                                macro_rules_binding.ident.name,
                                res,
                            ))
                        }
                    }
                }
                Scope::CrateRoot => {
                    let root_ident = Ident::new(kw::PathRoot, ident.span);
                    let root_module = this.resolve_crate_root(root_ident);
                    this.add_module_candidates(root_module, &mut suggestions, filter_fn);
                }
                Scope::Module(module, _) => {
                    this.add_module_candidates(module, &mut suggestions, filter_fn);
                }
                Scope::RegisteredAttrs => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Registered);
                    if filter_fn(res) {
                        suggestions.extend(
                            this.registered_attrs
                                .iter()
                                .map(|ident| TypoSuggestion::typo_from_res(ident.name, res)),
                        );
                    }
                }
                Scope::MacroUsePrelude => {
                    suggestions.extend(this.macro_use_prelude.iter().filter_map(
                        |(name, binding)| {
                            let res = binding.res();
                            filter_fn(res).then_some(TypoSuggestion::typo_from_res(*name, res))
                        },
                    ));
                }
                Scope::BuiltinAttrs => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Builtin(kw::Empty));
                    if filter_fn(res) {
                        suggestions.extend(
                            BUILTIN_ATTRIBUTES
                                .iter()
                                .map(|(name, ..)| TypoSuggestion::typo_from_res(*name, res)),
                        );
                    }
                }
                Scope::ExternPrelude => {
                    suggestions.extend(this.extern_prelude.iter().filter_map(|(ident, _)| {
                        let res = Res::Def(DefKind::Mod, DefId::local(CRATE_DEF_INDEX));
                        filter_fn(res).then_some(TypoSuggestion::typo_from_res(ident.name, res))
                    }));
                }
                Scope::ToolPrelude => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                    suggestions.extend(
                        this.registered_tools
                            .iter()
                            .map(|ident| TypoSuggestion::typo_from_res(ident.name, res)),
                    );
                }
                Scope::StdLibPrelude => {
                    if let Some(prelude) = this.prelude {
                        let mut tmp_suggestions = Vec::new();
                        this.add_module_candidates(prelude, &mut tmp_suggestions, filter_fn);
                        suggestions.extend(
                            tmp_suggestions
                                .into_iter()
                                .filter(|s| use_prelude || this.is_builtin_macro(s.res)),
                        );
                    }
                }
                Scope::BuiltinTypes => {
                    suggestions.extend(PrimTy::ALL.iter().filter_map(|prim_ty| {
                        let res = Res::PrimTy(*prim_ty);
                        filter_fn(res).then_some(TypoSuggestion::typo_from_res(prim_ty.name(), res))
                    }))
                }
            }

            None::<()>
        });

        // Make sure error reporting is deterministic.
        suggestions.sort_by_cached_key(|suggestion| suggestion.candidate.as_str());

        match find_best_match_for_name(
            &suggestions.iter().map(|suggestion| suggestion.candidate).collect::<Vec<Symbol>>(),
            ident.name,
            None,
        ) {
            Some(found) if found != ident.name => {
                suggestions.into_iter().find(|suggestion| suggestion.candidate == found)
            }
            _ => None,
        }
    }

    fn lookup_import_candidates_from_module<FilterFn>(
        &mut self,
        lookup_ident: Ident,
        namespace: Namespace,
        parent_scope: &ParentScope<'a>,
        start_module: Module<'a>,
        crate_name: Ident,
        filter_fn: FilterFn,
    ) -> Vec<ImportSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        let mut candidates = Vec::new();
        let mut seen_modules = FxHashSet::default();
        let mut worklist = vec![(start_module, Vec::<ast::PathSegment>::new(), true)];
        let mut worklist_via_import = vec![];

        while let Some((in_module, path_segments, accessible)) = match worklist.pop() {
            None => worklist_via_import.pop(),
            Some(x) => Some(x),
        } {
            let in_module_is_extern = !in_module.def_id().unwrap().is_local();
            // We have to visit module children in deterministic order to avoid
            // instabilities in reported imports (#43552).
            in_module.for_each_child(self, |this, ident, ns, name_binding| {
                // avoid non-importable candidates
                if !name_binding.is_importable() {
                    return;
                }

                let child_accessible =
                    accessible && this.is_accessible_from(name_binding.vis, parent_scope.module);

                // do not venture inside inaccessible items of other crates
                if in_module_is_extern && !child_accessible {
                    return;
                }

                let via_import = name_binding.is_import() && !name_binding.is_extern_crate();

                // There is an assumption elsewhere that paths of variants are in the enum's
                // declaration and not imported. With this assumption, the variant component is
                // chopped and the rest of the path is assumed to be the enum's own path. For
                // errors where a variant is used as the type instead of the enum, this causes
                // funny looking invalid suggestions, i.e `foo` instead of `foo::MyEnum`.
                if via_import && name_binding.is_possibly_imported_variant() {
                    return;
                }

                // collect results based on the filter function
                // avoid suggesting anything from the same module in which we are resolving
                if ident.name == lookup_ident.name
                    && ns == namespace
                    && !ptr::eq(in_module, parent_scope.module)
                {
                    let res = name_binding.res();
                    if filter_fn(res) {
                        // create the path
                        let mut segms = path_segments.clone();
                        if lookup_ident.span.rust_2018() {
                            // crate-local absolute paths start with `crate::` in edition 2018
                            // FIXME: may also be stabilized for Rust 2015 (Issues #45477, #44660)
                            segms.insert(0, ast::PathSegment::from_ident(crate_name));
                        }

                        segms.push(ast::PathSegment::from_ident(ident));
                        let path = Path { span: name_binding.span, segments: segms, tokens: None };
                        let did = match res {
                            Res::Def(DefKind::Ctor(..), did) => this.parent(did),
                            _ => res.opt_def_id(),
                        };

                        if child_accessible {
                            // Remove invisible match if exists
                            if let Some(idx) = candidates
                                .iter()
                                .position(|v: &ImportSuggestion| v.did == did && !v.accessible)
                            {
                                candidates.remove(idx);
                            }
                        }

                        if candidates.iter().all(|v: &ImportSuggestion| v.did != did) {
                            candidates.push(ImportSuggestion {
                                did,
                                descr: res.descr(),
                                path,
                                accessible: child_accessible,
                            });
                        }
                    }
                }

                // collect submodules to explore
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));

                    let is_extern_crate_that_also_appears_in_prelude =
                        name_binding.is_extern_crate() && lookup_ident.span.rust_2018();

                    if !is_extern_crate_that_also_appears_in_prelude {
                        // add the module to the lookup
                        if seen_modules.insert(module.def_id().unwrap()) {
                            if via_import { &mut worklist_via_import } else { &mut worklist }
                                .push((module, path_segments, child_accessible));
                        }
                    }
                }
            })
        }

        // If only some candidates are accessible, take just them
        if !candidates.iter().all(|v: &ImportSuggestion| !v.accessible) {
            candidates = candidates.into_iter().filter(|x| x.accessible).collect();
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
        &mut self,
        lookup_ident: Ident,
        namespace: Namespace,
        parent_scope: &ParentScope<'a>,
        filter_fn: FilterFn,
    ) -> Vec<ImportSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        let mut suggestions = self.lookup_import_candidates_from_module(
            lookup_ident,
            namespace,
            parent_scope,
            self.graph_root,
            Ident::with_dummy_span(kw::Crate),
            &filter_fn,
        );

        if lookup_ident.span.rust_2018() {
            let extern_prelude_names = self.extern_prelude.clone();
            for (ident, _) in extern_prelude_names.into_iter() {
                if ident.span.from_expansion() {
                    // Idents are adjusted to the root context before being
                    // resolved in the extern prelude, so reporting this to the
                    // user is no help. This skips the injected
                    // `extern crate std` in the 2018 edition, which would
                    // otherwise cause duplicate suggestions.
                    continue;
                }
                if let Some(crate_id) = self.crate_loader.maybe_process_path_extern(ident.name) {
                    let crate_root = self.expect_module(crate_id.as_def_id());
                    suggestions.extend(self.lookup_import_candidates_from_module(
                        lookup_ident,
                        namespace,
                        parent_scope,
                        crate_root,
                        ident,
                        &filter_fn,
                    ));
                }
            }
        }

        suggestions
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
            ScopeSet::Macro(macro_kind),
            parent_scope,
            ident,
            is_expected,
        );
        self.add_typo_suggestion(err, suggestion, ident.span);

        let import_suggestions =
            self.lookup_import_candidates(ident, Namespace::MacroNS, parent_scope, is_expected);
        show_candidates(
            &self.definitions,
            self.session,
            err,
            None,
            &import_suggestions,
            false,
            true,
        );

        if macro_kind == MacroKind::Derive && (ident.name == sym::Send || ident.name == sym::Sync) {
            let msg = format!("unsafe traits like `{}` should be implemented explicitly", ident);
            err.span_note(ident.span, &msg);
            return;
        }
        if self.macro_names.contains(&ident.normalize_to_macros_2_0()) {
            err.help("have you added the `#[macro_use]` on the module/import?");
            return;
        }
        for ns in [Namespace::MacroNS, Namespace::TypeNS, Namespace::ValueNS] {
            if let Ok(binding) = self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::All(ns, false),
                &parent_scope,
                false,
                false,
                ident.span,
            ) {
                let desc = match binding.res() {
                    Res::Def(DefKind::Macro(MacroKind::Bang), _) => {
                        "a function-like macro".to_string()
                    }
                    Res::Def(DefKind::Macro(MacroKind::Attr), _) | Res::NonMacroAttr(..) => {
                        format!("an attribute: `#[{}]`", ident)
                    }
                    Res::Def(DefKind::Macro(MacroKind::Derive), _) => {
                        format!("a derive macro: `#[derive({})]`", ident)
                    }
                    Res::ToolMod => {
                        // Don't confuse the user with tool modules.
                        continue;
                    }
                    Res::Def(DefKind::Trait, _) if macro_kind == MacroKind::Derive => {
                        "only a trait, without a derive macro".to_string()
                    }
                    res => format!(
                        "{} {}, not {} {}",
                        res.article(),
                        res.descr(),
                        macro_kind.article(),
                        macro_kind.descr_expected(),
                    ),
                };
                if let crate::NameBindingKind::Import { import, .. } = binding.kind {
                    if !import.span.is_dummy() {
                        err.span_note(
                            import.span,
                            &format!("`{}` is imported here, but it is {}", ident, desc),
                        );
                        // Silence the 'unused import' warning we might get,
                        // since this diagnostic already covers that import.
                        self.record_use(ident, binding, false);
                        return;
                    }
                }
                err.note(&format!("`{}` is in scope, but it is {}", ident, desc));
                return;
            }
        }
    }

    crate fn add_typo_suggestion(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        suggestion: Option<TypoSuggestion>,
        span: Span,
    ) -> bool {
        let suggestion = match suggestion {
            None => return false,
            // We shouldn't suggest underscore.
            Some(suggestion) if suggestion.candidate == kw::Underscore => return false,
            Some(suggestion) => suggestion,
        };
        let def_span = suggestion.res.opt_def_id().and_then(|def_id| match def_id.krate {
            LOCAL_CRATE => self.opt_span(def_id),
            _ => Some(
                self.session
                    .source_map()
                    .guess_head_span(self.cstore().get_span_untracked(def_id, self.session)),
            ),
        });
        if let Some(def_span) = def_span {
            if span.overlaps(def_span) {
                // Don't suggest typo suggestion for itself like in the following:
                // error[E0423]: expected function, tuple struct or tuple variant, found struct `X`
                //   --> $DIR/issue-64792-bad-unicode-ctor.rs:3:14
                //    |
                // LL | struct X {}
                //    | ----------- `X` defined here
                // LL |
                // LL | const Y: X = X("ö");
                //    | -------------^^^^^^- similarly named constant `Y` defined here
                //    |
                // help: use struct literal syntax instead
                //    |
                // LL | const Y: X = X {};
                //    |              ^^^^
                // help: a constant with a similar name exists
                //    |
                // LL | const Y: X = Y("ö");
                //    |              ^
                return false;
            }
            let prefix = match suggestion.target {
                SuggestionTarget::SimilarlyNamed => "similarly named ",
                SuggestionTarget::SingleItem => "",
            };

            err.span_label(
                self.session.source_map().guess_head_span(def_span),
                &format!(
                    "{}{} `{}` defined here",
                    prefix,
                    suggestion.res.descr(),
                    suggestion.candidate.as_str(),
                ),
            );
        }
        let msg = match suggestion.target {
            SuggestionTarget::SimilarlyNamed => format!(
                "{} {} with a similar name exists",
                suggestion.res.article(),
                suggestion.res.descr()
            ),
            SuggestionTarget::SingleItem => {
                format!("maybe you meant this {}", suggestion.res.descr())
            }
        };
        err.span_suggestion(
            span,
            &msg,
            suggestion.candidate.to_string(),
            Applicability::MaybeIncorrect,
        );
        true
    }

    fn binding_description(&self, b: &NameBinding<'_>, ident: Ident, from_prelude: bool) -> String {
        let res = b.res();
        if b.span.is_dummy() {
            // These already contain the "built-in" prefix or look bad with it.
            let add_built_in =
                !matches!(b.res(), Res::NonMacroAttr(..) | Res::PrimTy(..) | Res::ToolMod);
            let (built_in, from) = if from_prelude {
                ("", " from prelude")
            } else if b.is_extern_crate()
                && !b.is_import()
                && self.session.opts.externs.get(&ident.as_str()).is_some()
            {
                ("", " passed with `--extern`")
            } else if add_built_in {
                (" built-in", "")
            } else {
                ("", "")
            };

            let a = if built_in.is_empty() { res.article() } else { "a" };
            format!("{a}{built_in} {thing}{from}", thing = res.descr())
        } else {
            let introduced = if b.is_import() { "imported" } else { "defined" };
            format!("the {thing} {introduced} here", thing = res.descr())
        }
    }

    crate fn report_ambiguity_error(&self, ambiguity_error: &AmbiguityError<'_>) {
        let AmbiguityError { kind, ident, b1, b2, misc1, misc2 } = *ambiguity_error;
        let (b1, b2, misc1, misc2, swapped) = if b2.span.is_dummy() && !b1.span.is_dummy() {
            // We have to print the span-less alternative first, otherwise formatting looks bad.
            (b2, b1, misc2, misc1, true)
        } else {
            (b1, b2, misc1, misc2, false)
        };

        let mut err = struct_span_err!(
            self.session,
            ident.span,
            E0659,
            "`{ident}` is ambiguous ({why})",
            why = kind.descr()
        );
        err.span_label(ident.span, "ambiguous name");

        let mut could_refer_to = |b: &NameBinding<'_>, misc: AmbiguityErrorMisc, also: &str| {
            let what = self.binding_description(b, ident, misc == AmbiguityErrorMisc::FromPrelude);
            let note_msg = format!("`{ident}` could{also} refer to {what}");

            let thing = b.res().descr();
            let mut help_msgs = Vec::new();
            if b.is_glob_import()
                && (kind == AmbiguityKind::GlobVsGlob
                    || kind == AmbiguityKind::GlobVsExpanded
                    || kind == AmbiguityKind::GlobVsOuter && swapped != also.is_empty())
            {
                help_msgs.push(format!(
                    "consider adding an explicit import of `{ident}` to disambiguate"
                ))
            }
            if b.is_extern_crate() && ident.span.rust_2018() {
                help_msgs.push(format!("use `::{ident}` to refer to this {thing} unambiguously"))
            }
            if misc == AmbiguityErrorMisc::SuggestCrate {
                help_msgs
                    .push(format!("use `crate::{ident}` to refer to this {thing} unambiguously"))
            } else if misc == AmbiguityErrorMisc::SuggestSelf {
                help_msgs
                    .push(format!("use `self::{ident}` to refer to this {thing} unambiguously"))
            }

            err.span_note(b.span, &note_msg);
            for (i, help_msg) in help_msgs.iter().enumerate() {
                let or = if i == 0 { "" } else { "or " };
                err.help(&format!("{}{}", or, help_msg));
            }
        };

        could_refer_to(b1, misc1, "");
        could_refer_to(b2, misc2, " also");
        err.emit();
    }

    /// If the binding refers to a tuple struct constructor with fields,
    /// returns the span of its fields.
    fn ctor_fields_span(&self, binding: &NameBinding<'_>) -> Option<Span> {
        if let NameBindingKind::Res(
            Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), ctor_def_id),
            _,
        ) = binding.kind
        {
            let def_id = self.parent(ctor_def_id).expect("no parent for a constructor");
            let fields = self.field_names.get(&def_id)?;
            return fields.iter().map(|name| name.span).reduce(Span::to); // None for `struct Foo()`
        }
        None
    }

    crate fn report_privacy_error(&self, privacy_error: &PrivacyError<'_>) {
        let PrivacyError { ident, binding, .. } = *privacy_error;

        let res = binding.res();
        let ctor_fields_span = self.ctor_fields_span(binding);
        let plain_descr = res.descr().to_string();
        let nonimport_descr =
            if ctor_fields_span.is_some() { plain_descr + " constructor" } else { plain_descr };
        let import_descr = nonimport_descr.clone() + " import";
        let get_descr =
            |b: &NameBinding<'_>| if b.is_import() { &import_descr } else { &nonimport_descr };

        // Print the primary message.
        let descr = get_descr(binding);
        let mut err =
            struct_span_err!(self.session, ident.span, E0603, "{} `{}` is private", descr, ident);
        err.span_label(ident.span, &format!("private {}", descr));
        if let Some(span) = ctor_fields_span {
            err.span_label(span, "a constructor is private if any of the fields is private");
        }

        // Print the whole import chain to make it easier to see what happens.
        let first_binding = binding;
        let mut next_binding = Some(binding);
        let mut next_ident = ident;
        while let Some(binding) = next_binding {
            let name = next_ident;
            next_binding = match binding.kind {
                _ if res == Res::Err => None,
                NameBindingKind::Import { binding, import, .. } => match import.kind {
                    _ if binding.span.is_dummy() => None,
                    ImportKind::Single { source, .. } => {
                        next_ident = source;
                        Some(binding)
                    }
                    ImportKind::Glob { .. } | ImportKind::MacroUse => Some(binding),
                    ImportKind::ExternCrate { .. } => None,
                },
                _ => None,
            };

            let first = ptr::eq(binding, first_binding);
            let msg = format!(
                "{and_refers_to}the {item} `{name}`{which} is defined here{dots}",
                and_refers_to = if first { "" } else { "...and refers to " },
                item = get_descr(binding),
                which = if first { "" } else { " which" },
                dots = if next_binding.is_some() { "..." } else { "" },
            );
            let def_span = self.session.source_map().guess_head_span(binding.span);
            let mut note_span = MultiSpan::from_span(def_span);
            if !first && binding.vis == ty::Visibility::Public {
                note_span.push_span_label(def_span, "consider importing it directly".into());
            }
            err.span_note(note_span, &msg);
        }

        err.emit();
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
            (Some(fst), Some(snd))
                if fst.ident.name == kw::PathRoot && !snd.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018.
            (Some(fst), _)
                if fst.ident.span.rust_2018() && !fst.ident.is_path_segment_keyword() =>
            {
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
    /// ```text
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
        let result = self.r.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_self_suggestion: path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result { Some((path, Vec::new())) } else { None }
    }

    /// Suggests a missing `crate::` if that resolves to an correct module.
    ///
    /// ```text
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
        let result = self.r.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_crate_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                vec![
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>"
                        .to_string(),
                ],
            ))
        } else {
            None
        }
    }

    /// Suggests a missing `super::` if that resolves to an correct module.
    ///
    /// ```text
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
        let result = self.r.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_super_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result { Some((path, Vec::new())) } else { None }
    }

    /// Suggests a missing external crate name if that resolves to an correct module.
    ///
    /// ```text
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
        // 1) some consistent ordering for emitted diagnostics, and
        // 2) `std` suggestions before `core` suggestions.
        let mut extern_crate_names =
            self.r.extern_prelude.iter().map(|(ident, _)| ident.name).collect::<Vec<_>>();
        extern_crate_names.sort_by_key(|name| Reverse(name.as_str()));

        for name in extern_crate_names.into_iter() {
            // Replace first ident with a crate name and check if that is valid.
            path[0].ident.name = name;
            let result = self.r.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
            debug!(
                "make_external_crate_suggestion: name={:?} path={:?} result={:?}",
                name, path, result
            );
            if let PathResult::Module(..) = result {
                return Some((path, Vec::new()));
            }
        }

        None
    }

    /// Suggests importing a macro from the root of the crate rather than a module within
    /// the crate.
    ///
    /// ```text
    /// help: a macro with this name exists at the root of the crate
    ///    |
    /// LL | use issue_59764::makro;
    ///    |     ^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: this could be because a macro annotated with `#[macro_export]` will be exported
    ///            at the root of the crate instead of the module where it is defined
    /// ```
    pub(crate) fn check_for_module_export_macro(
        &mut self,
        import: &'b Import<'b>,
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

        let resolutions = self.r.resolutions(crate_module).borrow();
        let resolution = resolutions.get(&self.r.new_key(ident, MacroNS))?;
        let binding = resolution.borrow().binding()?;
        if let Res::Def(DefKind::Macro(MacroKind::Bang), _) = binding.res() {
            let module_name = crate_module.kind.name().unwrap();
            let import_snippet = match import.kind {
                ImportKind::Single { source, target, .. } if source != target => {
                    format!("{} as {}", source, target)
                }
                _ => format!("{}", ident),
            };

            let mut corrections: Vec<(Span, String)> = Vec::new();
            if !import.is_nested() {
                // Assume this is the easy case of `use issue_59764::foo::makro;` and just remove
                // intermediate segments.
                corrections.push((import.span, format!("{}::{}", module_name, import_snippet)));
            } else {
                // Find the binding span (and any trailing commas and spaces).
                //   ie. `use a::b::{c, d, e};`
                //                      ^^^
                let (found_closing_brace, binding_span) = find_span_of_binding_until_next_binding(
                    self.r.session,
                    import.span,
                    import.use_span,
                );
                debug!(
                    "check_for_module_export_macro: found_closing_brace={:?} binding_span={:?}",
                    found_closing_brace, binding_span
                );

                let mut removal_span = binding_span;
                if found_closing_brace {
                    // If the binding span ended with a closing brace, as in the below example:
                    //   ie. `use a::b::{c, d};`
                    //                      ^
                    // Then expand the span of characters to remove to include the previous
                    // binding's trailing comma.
                    //   ie. `use a::b::{c, d};`
                    //                    ^^^
                    if let Some(previous_span) =
                        extend_span_to_previous_binding(self.r.session, binding_span)
                    {
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
                    self.r.session,
                    module_name,
                    import.use_span,
                );
                debug!(
                    "check_for_module_export_macro: has_nested={:?} after_crate_name={:?}",
                    has_nested, after_crate_name
                );

                let source_map = self.r.session.source_map();

                // Add the import to the start, with a `{` if required.
                let start_point = source_map.start_point(after_crate_name);
                if let Ok(start_snippet) = source_map.span_to_snippet(start_point) {
                    corrections.push((
                        start_point,
                        if has_nested {
                            // In this case, `start_snippet` must equal '{'.
                            format!("{}{}, ", start_snippet, import_snippet)
                        } else {
                            // In this case, add a `{`, then the moved import, then whatever
                            // was there before.
                            format!("{{{}, {}", import_snippet, start_snippet)
                        },
                    ));
                }

                // Add a `};` to the end if nested, matching the `{` added at the start.
                if !has_nested {
                    corrections.push((source_map.end_point(after_crate_name), "};".to_string()));
                }
            }

            let suggestion = Some((
                corrections,
                String::from("a macro with this name exists at the root of the crate"),
                Applicability::MaybeIncorrect,
            ));
            let note = vec![
                "this could be because a macro annotated with `#[macro_export]` will be exported \
                 at the root of the crate instead of the module where it is defined"
                    .to_string(),
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
    let after_binding_until_next_binding =
        source_map.span_take_while(after_binding_until_end, |&ch| {
            if ch == '}' {
                found_closing_brace = true;
            }
            ch == ' ' || ch == ','
        });

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
pub(crate) fn extend_span_to_previous_binding(sess: &Session, binding_span: Span) -> Option<Span> {
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
        binding_span.lo().0 - (prev_comma.as_bytes().len() as u32) - 1,
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
    debug!(
        "find_span_immediately_after_crate_name: module_name={:?} use_span={:?}",
        module_name, use_span
    );
    let source_map = sess.source_map();

    // Using `use issue_59764::foo::{baz, makro};` as an example throughout..
    let mut num_colons = 0;
    // Find second colon.. `use issue_59764:`
    let until_second_colon = source_map.span_take_while(use_span, |c| {
        if *c == ':' {
            num_colons += 1;
        }
        !matches!(c, ':' if num_colons == 2)
    });
    // Find everything after the second colon.. `foo::{baz, makro};`
    let from_second_colon = use_span.with_lo(until_second_colon.hi() + BytePos(1));

    let mut found_a_non_whitespace_character = false;
    // Find the first non-whitespace character in `from_second_colon`.. `f`
    let after_second_colon = source_map.span_take_while(from_second_colon, |c| {
        if found_a_non_whitespace_character {
            return false;
        }
        if !c.is_whitespace() {
            found_a_non_whitespace_character = true;
        }
        true
    });

    // Find the first `{` in from_second_colon.. `foo::{`
    let next_left_bracket = source_map.span_through_char(from_second_colon, '{');

    (next_left_bracket == after_second_colon, from_second_colon)
}

/// When an entity with a given name is not available in scope, we search for
/// entities with that name in all crates. This method allows outputting the
/// results of this search in a programmer-friendly way
crate fn show_candidates(
    definitions: &rustc_hir::definitions::Definitions,
    session: &Session,
    err: &mut DiagnosticBuilder<'_>,
    // This is `None` if all placement locations are inside expansions
    use_placement_span: Option<Span>,
    candidates: &[ImportSuggestion],
    instead: bool,
    found_use: bool,
) {
    if candidates.is_empty() {
        return;
    }

    let mut accessible_path_strings: Vec<(String, &str, Option<DefId>)> = Vec::new();
    let mut inaccessible_path_strings: Vec<(String, &str, Option<DefId>)> = Vec::new();

    candidates.iter().for_each(|c| {
        (if c.accessible { &mut accessible_path_strings } else { &mut inaccessible_path_strings })
            .push((path_names_to_string(&c.path), c.descr, c.did))
    });

    // we want consistent results across executions, but candidates are produced
    // by iterating through a hash map, so make sure they are ordered:
    for path_strings in [&mut accessible_path_strings, &mut inaccessible_path_strings] {
        path_strings.sort_by(|a, b| a.0.cmp(&b.0));
        let core_path_strings =
            path_strings.drain_filter(|p| p.0.starts_with("core::")).collect::<Vec<_>>();
        path_strings.extend(core_path_strings);
        path_strings.dedup_by(|a, b| a.0 == b.0);
    }

    if !accessible_path_strings.is_empty() {
        let (determiner, kind) = if accessible_path_strings.len() == 1 {
            ("this", accessible_path_strings[0].1)
        } else {
            ("one of these", "items")
        };

        let instead = if instead { " instead" } else { "" };
        let mut msg = format!("consider importing {} {}{}", determiner, kind, instead);

        if let Some(span) = use_placement_span {
            for candidate in &mut accessible_path_strings {
                // produce an additional newline to separate the new use statement
                // from the directly following item.
                let additional_newline = if found_use { "" } else { "\n" };
                candidate.0 = format!("use {};\n{}", &candidate.0, additional_newline);
            }

            err.span_suggestions(
                span,
                &msg,
                accessible_path_strings.into_iter().map(|a| a.0),
                Applicability::Unspecified,
            );
        } else {
            msg.push(':');

            for candidate in accessible_path_strings {
                msg.push('\n');
                msg.push_str(&candidate.0);
            }

            err.note(&msg);
        }
    } else {
        assert!(!inaccessible_path_strings.is_empty());

        if inaccessible_path_strings.len() == 1 {
            let (name, descr, def_id) = &inaccessible_path_strings[0];
            let msg = format!("{} `{}` exists but is inaccessible", descr, name);

            if let Some(local_def_id) = def_id.and_then(|did| did.as_local()) {
                let span = definitions.def_span(local_def_id);
                let span = session.source_map().guess_head_span(span);
                let mut multi_span = MultiSpan::from_span(span);
                multi_span.push_span_label(span, "not accessible".to_string());
                err.span_note(multi_span, &msg);
            } else {
                err.note(&msg);
            }
        } else {
            let (_, descr_first, _) = &inaccessible_path_strings[0];
            let descr = if inaccessible_path_strings
                .iter()
                .skip(1)
                .all(|(_, descr, _)| descr == descr_first)
            {
                format!("{}", descr_first)
            } else {
                "item".to_string()
            };

            let mut msg = format!("these {}s exist but are inaccessible", descr);
            let mut has_colon = false;

            let mut spans = Vec::new();
            for (name, _, def_id) in &inaccessible_path_strings {
                if let Some(local_def_id) = def_id.and_then(|did| did.as_local()) {
                    let span = definitions.def_span(local_def_id);
                    let span = session.source_map().guess_head_span(span);
                    spans.push((name, span));
                } else {
                    if !has_colon {
                        msg.push(':');
                        has_colon = true;
                    }
                    msg.push('\n');
                    msg.push_str(name);
                }
            }

            let mut multi_span = MultiSpan::from_spans(spans.iter().map(|(_, sp)| *sp).collect());
            for (name, span) in spans {
                multi_span.push_span_label(span, format!("`{}`: not accessible", name));
            }

            err.span_note(multi_span, &msg);
        }
    }
}
