use rustc_ast::expand::StrippedCfgItem;
use rustc_ast::ptr::P;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{
    self as ast, CRATE_NODE_ID, Crate, ItemKind, MetaItemInner, MetaItemKind, ModKind, NodeId, Path,
};
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::{self as attr, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::unord::UnordSet;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, ErrorGuaranteed, MultiSpan, SuggestionStyle,
    report_ambiguity_error, struct_span_code_err,
};
use rustc_feature::BUILTIN_ATTRIBUTES;
use rustc_hir::PrimTy;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, CtorOf, DefKind, NonMacroAttrKind, PerNS};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::builtin::{
    ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE, AMBIGUOUS_GLOB_IMPORTS,
    MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
};
use rustc_session::lint::{AmbiguityErrorDiag, BuiltinLintDiag};
use rustc_session::utils::was_invoked_from_cargo;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::edition::Edition;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Ident, Span, Symbol, SyntaxContext, kw, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, instrument};

use crate::errors::{
    self, AddedMacroUse, ChangeImportBinding, ChangeImportBindingSuggestion, ConsiderAddingADerive,
    ExplicitUnsafeTraits, MacroDefinedLater, MacroRulesNot, MacroSuggMovePosition,
    MaybeMissingMacroRulesName,
};
use crate::imports::{Import, ImportKind};
use crate::late::{PatternSource, Rib};
use crate::{
    AmbiguityError, AmbiguityErrorMisc, AmbiguityKind, BindingError, BindingKey, Finalize,
    ForwardGenericParamBanReason, HasGenericParams, LexicalScopeBinding, MacroRulesScope, Module,
    ModuleKind, ModuleOrUniformRoot, NameBinding, NameBindingKind, ParentScope, PathResult,
    PrivacyError, ResolutionError, Resolver, Scope, ScopeSet, Segment, UseError, Used,
    VisResolutionError, errors as errs, path_names_to_string,
};

type Res = def::Res<ast::NodeId>;

/// A vector of spans and replacements, a message and applicability.
pub(crate) type Suggestion = (Vec<(Span, String)>, String, Applicability);

/// Potential candidate for an undeclared or out-of-scope label - contains the ident of a
/// similarly named label and whether or not it is reachable.
pub(crate) type LabelSuggestion = (Ident, bool);

#[derive(Debug)]
pub(crate) enum SuggestionTarget {
    /// The target has a similar name as the name used by the programmer (probably a typo)
    SimilarlyNamed,
    /// The target is the only valid item that can be used in the corresponding context
    SingleItem,
}

#[derive(Debug)]
pub(crate) struct TypoSuggestion {
    pub candidate: Symbol,
    /// The source location where the name is defined; None if the name is not defined
    /// in source e.g. primitives
    pub span: Option<Span>,
    pub res: Res,
    pub target: SuggestionTarget,
}

impl TypoSuggestion {
    pub(crate) fn typo_from_ident(ident: Ident, res: Res) -> TypoSuggestion {
        Self {
            candidate: ident.name,
            span: Some(ident.span),
            res,
            target: SuggestionTarget::SimilarlyNamed,
        }
    }
    pub(crate) fn typo_from_name(candidate: Symbol, res: Res) -> TypoSuggestion {
        Self { candidate, span: None, res, target: SuggestionTarget::SimilarlyNamed }
    }
    pub(crate) fn single_item_from_ident(ident: Ident, res: Res) -> TypoSuggestion {
        Self {
            candidate: ident.name,
            span: Some(ident.span),
            res,
            target: SuggestionTarget::SingleItem,
        }
    }
}

/// A free importable items suggested in case of resolution failure.
#[derive(Debug, Clone)]
pub(crate) struct ImportSuggestion {
    pub did: Option<DefId>,
    pub descr: &'static str,
    pub path: Path,
    pub accessible: bool,
    // false if the path traverses a foreign `#[doc(hidden)]` item.
    pub doc_visible: bool,
    pub via_import: bool,
    /// An extra note that should be issued if this item is suggested
    pub note: Option<String>,
    pub is_stable: bool,
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

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }

    pub(crate) fn report_errors(&mut self, krate: &Crate) {
        self.report_with_use_injections(krate);

        for &(span_use, span_def) in &self.macro_expanded_macro_export_errors {
            self.lint_buffer.buffer_lint(
                MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
                CRATE_NODE_ID,
                span_use,
                BuiltinLintDiag::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def),
            );
        }

        for ambiguity_error in &self.ambiguity_errors {
            let diag = self.ambiguity_diagnostics(ambiguity_error);
            if ambiguity_error.warning {
                let NameBindingKind::Import { import, .. } = ambiguity_error.b1.0.kind else {
                    unreachable!()
                };
                self.lint_buffer.buffer_lint(
                    AMBIGUOUS_GLOB_IMPORTS,
                    import.root_id,
                    ambiguity_error.ident.span,
                    BuiltinLintDiag::AmbiguousGlobImports { diag },
                );
            } else {
                let mut err = struct_span_code_err!(self.dcx(), diag.span, E0659, "{}", diag.msg);
                report_ambiguity_error(&mut err, diag);
                err.emit();
            }
        }

        let mut reported_spans = FxHashSet::default();
        for error in std::mem::take(&mut self.privacy_errors) {
            if reported_spans.insert(error.dedup_span) {
                self.report_privacy_error(&error);
            }
        }
    }

    fn report_with_use_injections(&mut self, krate: &Crate) {
        for UseError { mut err, candidates, def_id, instead, suggestion, path, is_call } in
            std::mem::take(&mut self.use_injections)
        {
            let (span, found_use) = if let Some(def_id) = def_id.as_local() {
                UsePlacementFinder::check(krate, self.def_id_to_node_id(def_id))
            } else {
                (None, FoundUse::No)
            };

            if !candidates.is_empty() {
                show_candidates(
                    self.tcx,
                    &mut err,
                    span,
                    &candidates,
                    if instead { Instead::Yes } else { Instead::No },
                    found_use,
                    DiagMode::Normal,
                    path,
                    "",
                );
                err.emit();
            } else if let Some((span, msg, sugg, appl)) = suggestion {
                err.span_suggestion_verbose(span, msg, sugg, appl);
                err.emit();
            } else if let [segment] = path.as_slice()
                && is_call
            {
                err.stash(segment.ident.span, rustc_errors::StashKey::CallIntoMethod);
            } else {
                err.emit();
            }
        }
    }

    pub(crate) fn report_conflict(
        &mut self,
        parent: Module<'_>,
        ident: Ident,
        ns: Namespace,
        new_binding: NameBinding<'ra>,
        old_binding: NameBinding<'ra>,
    ) {
        // Error on the second of two conflicting names
        if old_binding.span.lo() > new_binding.span.lo() {
            return self.report_conflict(parent, ident, ns, old_binding, new_binding);
        }

        let container = match parent.kind {
            // Avoid using TyCtxt::def_kind_descr in the resolver, because it
            // indirectly *calls* the resolver, and would cause a query cycle.
            ModuleKind::Def(kind, _, _) => kind.descr(parent.def_id()),
            ModuleKind::Block => "block",
        };

        let (name, span) =
            (ident.name, self.tcx.sess.source_map().guess_head_span(new_binding.span));

        if self.name_already_seen.get(&name) == Some(&span) {
            return;
        }

        let old_kind = match (ns, old_binding.module()) {
            (ValueNS, _) => "value",
            (MacroNS, _) => "macro",
            (TypeNS, _) if old_binding.is_extern_crate() => "extern crate",
            (TypeNS, Some(module)) if module.is_normal() => "module",
            (TypeNS, Some(module)) if module.is_trait() => "trait",
            (TypeNS, _) => "type",
        };

        let code = match (old_binding.is_extern_crate(), new_binding.is_extern_crate()) {
            (true, true) => E0259,
            (true, _) | (_, true) => match new_binding.is_import() && old_binding.is_import() {
                true => E0254,
                false => E0260,
            },
            _ => match (old_binding.is_import_user_facing(), new_binding.is_import_user_facing()) {
                (false, false) => E0428,
                (true, true) => E0252,
                _ => E0255,
            },
        };

        let label = match new_binding.is_import_user_facing() {
            true => errors::NameDefinedMultipleTimeLabel::Reimported { span, name },
            false => errors::NameDefinedMultipleTimeLabel::Redefined { span, name },
        };

        let old_binding_label =
            (!old_binding.span.is_dummy() && old_binding.span != span).then(|| {
                let span = self.tcx.sess.source_map().guess_head_span(old_binding.span);
                match old_binding.is_import_user_facing() {
                    true => errors::NameDefinedMultipleTimeOldBindingLabel::Import {
                        span,
                        name,
                        old_kind,
                    },
                    false => errors::NameDefinedMultipleTimeOldBindingLabel::Definition {
                        span,
                        name,
                        old_kind,
                    },
                }
            });

        let mut err = self
            .dcx()
            .create_err(errors::NameDefinedMultipleTime {
                span,
                descr: ns.descr(),
                container,
                label,
                old_binding_label,
            })
            .with_code(code);

        // See https://github.com/rust-lang/rust/issues/32354
        use NameBindingKind::Import;
        let can_suggest = |binding: NameBinding<'_>, import: self::Import<'_>| {
            !binding.span.is_dummy()
                && !matches!(import.kind, ImportKind::MacroUse { .. } | ImportKind::MacroExport)
        };
        let import = match (&new_binding.kind, &old_binding.kind) {
            // If there are two imports where one or both have attributes then prefer removing the
            // import without attributes.
            (Import { import: new, .. }, Import { import: old, .. })
                if {
                    (new.has_attributes || old.has_attributes)
                        && can_suggest(old_binding, *old)
                        && can_suggest(new_binding, *new)
                } =>
            {
                if old.has_attributes {
                    Some((*new, new_binding.span, true))
                } else {
                    Some((*old, old_binding.span, true))
                }
            }
            // Otherwise prioritize the new binding.
            (Import { import, .. }, other) if can_suggest(new_binding, *import) => {
                Some((*import, new_binding.span, other.is_import()))
            }
            (other, Import { import, .. }) if can_suggest(old_binding, *import) => {
                Some((*import, old_binding.span, other.is_import()))
            }
            _ => None,
        };

        // Check if the target of the use for both bindings is the same.
        let duplicate = new_binding.res().opt_def_id() == old_binding.res().opt_def_id();
        let has_dummy_span = new_binding.span.is_dummy() || old_binding.span.is_dummy();
        let from_item =
            self.extern_prelude.get(&ident).is_none_or(|entry| entry.introduced_by_item);
        // Only suggest removing an import if both bindings are to the same def, if both spans
        // aren't dummy spans. Further, if both bindings are imports, then the ident must have
        // been introduced by an item.
        let should_remove_import = duplicate
            && !has_dummy_span
            && ((new_binding.is_extern_crate() || old_binding.is_extern_crate()) || from_item);

        match import {
            Some((import, span, true)) if should_remove_import && import.is_nested() => {
                self.add_suggestion_for_duplicate_nested_use(&mut err, import, span);
            }
            Some((import, _, true)) if should_remove_import && !import.is_glob() => {
                // Simple case - remove the entire import. Due to the above match arm, this can
                // only be a single use so just remove it entirely.
                err.subdiagnostic(errors::ToolOnlyRemoveUnnecessaryImport {
                    span: import.use_span_with_attributes,
                });
            }
            Some((import, span, _)) => {
                self.add_suggestion_for_rename_of_use(&mut err, name, import, span);
            }
            _ => {}
        }

        err.emit();
        self.name_already_seen.insert(name, span);
    }

    /// This function adds a suggestion to change the binding name of a new import that conflicts
    /// with an existing import.
    ///
    /// ```text,ignore (diagnostic)
    /// help: you can use `as` to change the binding name of the import
    ///    |
    /// LL | use foo::bar as other_bar;
    ///    |     ^^^^^^^^^^^^^^^^^^^^^
    /// ```
    fn add_suggestion_for_rename_of_use(
        &self,
        err: &mut Diag<'_>,
        name: Symbol,
        import: Import<'_>,
        binding_span: Span,
    ) {
        let suggested_name = if name.as_str().chars().next().unwrap().is_uppercase() {
            format!("Other{name}")
        } else {
            format!("other_{name}")
        };

        let mut suggestion = None;
        let mut span = binding_span;
        match import.kind {
            ImportKind::Single { type_ns_only: true, .. } => {
                suggestion = Some(format!("self as {suggested_name}"))
            }
            ImportKind::Single { source, .. } => {
                if let Some(pos) = source.span.hi().0.checked_sub(binding_span.lo().0)
                    && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(binding_span)
                    && pos as usize <= snippet.len()
                {
                    span = binding_span.with_lo(binding_span.lo() + BytePos(pos)).with_hi(
                        binding_span.hi() - BytePos(if snippet.ends_with(';') { 1 } else { 0 }),
                    );
                    suggestion = Some(format!(" as {suggested_name}"));
                }
            }
            ImportKind::ExternCrate { source, target, .. } => {
                suggestion = Some(format!(
                    "extern crate {} as {};",
                    source.unwrap_or(target.name),
                    suggested_name,
                ))
            }
            _ => unreachable!(),
        }

        if let Some(suggestion) = suggestion {
            err.subdiagnostic(ChangeImportBindingSuggestion { span, suggestion });
        } else {
            err.subdiagnostic(ChangeImportBinding { span });
        }
    }

    /// This function adds a suggestion to remove an unnecessary binding from an import that is
    /// nested. In the following example, this function will be invoked to remove the `a` binding
    /// in the second use statement:
    ///
    /// ```ignore (diagnostic)
    /// use issue_52891::a;
    /// use issue_52891::{d, a, e};
    /// ```
    ///
    /// The following suggestion will be added:
    ///
    /// ```ignore (diagnostic)
    /// use issue_52891::{d, a, e};
    ///                      ^-- help: remove unnecessary import
    /// ```
    ///
    /// If the nested use contains only one import then the suggestion will remove the entire
    /// line.
    ///
    /// It is expected that the provided import is nested - this isn't checked by the
    /// function. If this invariant is not upheld, this function's behaviour will be unexpected
    /// as characters expected by span manipulations won't be present.
    fn add_suggestion_for_duplicate_nested_use(
        &self,
        err: &mut Diag<'_>,
        import: Import<'_>,
        binding_span: Span,
    ) {
        assert!(import.is_nested());

        // Two examples will be used to illustrate the span manipulations we're doing:
        //
        // - Given `use issue_52891::{d, a, e};` where `a` is a duplicate then `binding_span` is
        //   `a` and `import.use_span` is `issue_52891::{d, a, e};`.
        // - Given `use issue_52891::{d, e, a};` where `a` is a duplicate then `binding_span` is
        //   `a` and `import.use_span` is `issue_52891::{d, e, a};`.

        let (found_closing_brace, span) =
            find_span_of_binding_until_next_binding(self.tcx.sess, binding_span, import.use_span);

        // If there was a closing brace then identify the span to remove any trailing commas from
        // previous imports.
        if found_closing_brace {
            if let Some(span) = extend_span_to_previous_binding(self.tcx.sess, span) {
                err.subdiagnostic(errors::ToolOnlyRemoveUnnecessaryImport { span });
            } else {
                // Remove the entire line if we cannot extend the span back, this indicates an
                // `issue_52891::{self}` case.
                err.subdiagnostic(errors::RemoveUnnecessaryImport {
                    span: import.use_span_with_attributes,
                });
            }

            return;
        }

        err.subdiagnostic(errors::RemoveUnnecessaryImport { span });
    }

    pub(crate) fn lint_if_path_starts_with_module(
        &mut self,
        finalize: Option<Finalize>,
        path: &[Segment],
        second_binding: Option<NameBinding<'_>>,
    ) {
        let Some(Finalize { node_id, root_span, .. }) = finalize else {
            return;
        };

        let first_name = match path.get(0) {
            // In the 2018 edition this lint is a hard error, so nothing to do
            Some(seg) if seg.ident.span.is_rust_2015() && self.tcx.sess.is_rust_2015() => {
                seg.ident.name
            }
            _ => return,
        };

        // We're only interested in `use` paths which should start with
        // `{{root}}` currently.
        if first_name != kw::PathRoot {
            return;
        }

        match path.get(1) {
            // If this import looks like `crate::...` it's already good
            Some(Segment { ident, .. }) if ident.name == kw::Crate => return,
            // Otherwise go below to see if it's an extern crate
            Some(_) => {}
            // If the path has length one (and it's `PathRoot` most likely)
            // then we don't know whether we're gonna be importing a crate or an
            // item in our crate. Defer this lint to elsewhere
            None => return,
        }

        // If the first element of our path was actually resolved to an
        // `ExternCrate` (also used for `crate::...`) then no need to issue a
        // warning, this looks all good!
        if let Some(binding) = second_binding
            && let NameBindingKind::Import { import, .. } = binding.kind
            // Careful: we still want to rewrite paths from renamed extern crates.
            && let ImportKind::ExternCrate { source: None, .. } = import.kind
        {
            return;
        }

        let diag = BuiltinLintDiag::AbsPathWithModule(root_span);
        self.lint_buffer.buffer_lint(
            ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
            node_id,
            root_span,
            diag,
        );
    }

    pub(crate) fn add_module_candidates(
        &mut self,
        module: Module<'ra>,
        names: &mut Vec<TypoSuggestion>,
        filter_fn: &impl Fn(Res) -> bool,
        ctxt: Option<SyntaxContext>,
    ) {
        module.for_each_child(self, |_this, ident, _ns, binding| {
            let res = binding.res();
            if filter_fn(res) && ctxt.is_none_or(|ctxt| ctxt == ident.span.ctxt()) {
                names.push(TypoSuggestion::typo_from_ident(ident, res));
            }
        });
    }

    /// Combines an error with provided span and emits it.
    ///
    /// This takes the error provided, combines it with the span and any additional spans inside the
    /// error and emits it.
    pub(crate) fn report_error(
        &mut self,
        span: Span,
        resolution_error: ResolutionError<'ra>,
    ) -> ErrorGuaranteed {
        self.into_struct_error(span, resolution_error).emit()
    }

    pub(crate) fn into_struct_error(
        &mut self,
        span: Span,
        resolution_error: ResolutionError<'ra>,
    ) -> Diag<'_> {
        match resolution_error {
            ResolutionError::GenericParamsFromOuterItem(
                outer_res,
                has_generic_params,
                def_kind,
            ) => {
                use errs::GenericParamsFromOuterItemLabel as Label;
                let static_or_const = match def_kind {
                    DefKind::Static { .. } => {
                        Some(errs::GenericParamsFromOuterItemStaticOrConst::Static)
                    }
                    DefKind::Const => Some(errs::GenericParamsFromOuterItemStaticOrConst::Const),
                    _ => None,
                };
                let is_self =
                    matches!(outer_res, Res::SelfTyParam { .. } | Res::SelfTyAlias { .. });
                let mut err = errs::GenericParamsFromOuterItem {
                    span,
                    label: None,
                    refer_to_type_directly: None,
                    sugg: None,
                    static_or_const,
                    is_self,
                };

                let sm = self.tcx.sess.source_map();
                let def_id = match outer_res {
                    Res::SelfTyParam { .. } => {
                        err.label = Some(Label::SelfTyParam(span));
                        return self.dcx().create_err(err);
                    }
                    Res::SelfTyAlias { alias_to: def_id, .. } => {
                        err.label = Some(Label::SelfTyAlias(reduce_impl_span_to_impl_keyword(
                            sm,
                            self.def_span(def_id),
                        )));
                        err.refer_to_type_directly = Some(span);
                        return self.dcx().create_err(err);
                    }
                    Res::Def(DefKind::TyParam, def_id) => {
                        err.label = Some(Label::TyParam(self.def_span(def_id)));
                        def_id
                    }
                    Res::Def(DefKind::ConstParam, def_id) => {
                        err.label = Some(Label::ConstParam(self.def_span(def_id)));
                        def_id
                    }
                    _ => {
                        bug!(
                            "GenericParamsFromOuterItem should only be used with \
                            Res::SelfTyParam, Res::SelfTyAlias, DefKind::TyParam or \
                            DefKind::ConstParam"
                        );
                    }
                };

                if let HasGenericParams::Yes(span) = has_generic_params {
                    let name = self.tcx.item_name(def_id);
                    let (span, snippet) = if span.is_empty() {
                        let snippet = format!("<{name}>");
                        (span, snippet)
                    } else {
                        let span = sm.span_through_char(span, '<').shrink_to_hi();
                        let snippet = format!("{name}, ");
                        (span, snippet)
                    };
                    err.sugg = Some(errs::GenericParamsFromOuterItemSugg { span, snippet });
                }

                self.dcx().create_err(err)
            }
            ResolutionError::NameAlreadyUsedInParameterList(name, first_use_span) => self
                .dcx()
                .create_err(errs::NameAlreadyUsedInParameterList { span, first_use_span, name }),
            ResolutionError::MethodNotMemberOfTrait(method, trait_, candidate) => {
                self.dcx().create_err(errs::MethodNotMemberOfTrait {
                    span,
                    method,
                    trait_,
                    sub: candidate.map(|c| errs::AssociatedFnWithSimilarNameExists {
                        span: method.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::TypeNotMemberOfTrait(type_, trait_, candidate) => {
                self.dcx().create_err(errs::TypeNotMemberOfTrait {
                    span,
                    type_,
                    trait_,
                    sub: candidate.map(|c| errs::AssociatedTypeWithSimilarNameExists {
                        span: type_.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::ConstNotMemberOfTrait(const_, trait_, candidate) => {
                self.dcx().create_err(errs::ConstNotMemberOfTrait {
                    span,
                    const_,
                    trait_,
                    sub: candidate.map(|c| errs::AssociatedConstWithSimilarNameExists {
                        span: const_.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::VariableNotBoundInPattern(binding_error, parent_scope) => {
                let BindingError { name, target, origin, could_be_path } = binding_error;

                let target_sp = target.iter().copied().collect::<Vec<_>>();
                let origin_sp = origin.iter().copied().collect::<Vec<_>>();

                let msp = MultiSpan::from_spans(target_sp.clone());
                let mut err = self
                    .dcx()
                    .create_err(errors::VariableIsNotBoundInAllPatterns { multispan: msp, name });
                for sp in target_sp {
                    err.subdiagnostic(errors::PatternDoesntBindName { span: sp, name });
                }
                for sp in origin_sp {
                    err.subdiagnostic(errors::VariableNotInAllPatterns { span: sp });
                }
                if could_be_path {
                    let import_suggestions = self.lookup_import_candidates(
                        name,
                        Namespace::ValueNS,
                        &parent_scope,
                        &|res: Res| {
                            matches!(
                                res,
                                Res::Def(
                                    DefKind::Ctor(CtorOf::Variant, CtorKind::Const)
                                        | DefKind::Ctor(CtorOf::Struct, CtorKind::Const)
                                        | DefKind::Const
                                        | DefKind::AssocConst,
                                    _,
                                )
                            )
                        },
                    );

                    if import_suggestions.is_empty() {
                        let help_msg = format!(
                            "if you meant to match on a variant or a `const` item, consider \
                             making the path in the pattern qualified: `path::to::ModOrType::{name}`",
                        );
                        err.span_help(span, help_msg);
                    }
                    show_candidates(
                        self.tcx,
                        &mut err,
                        Some(span),
                        &import_suggestions,
                        Instead::No,
                        FoundUse::Yes,
                        DiagMode::Pattern,
                        vec![],
                        "",
                    );
                }
                err
            }
            ResolutionError::VariableBoundWithDifferentMode(variable_name, first_binding_span) => {
                self.dcx().create_err(errs::VariableBoundWithDifferentMode {
                    span,
                    first_binding_span,
                    variable_name,
                })
            }
            ResolutionError::IdentifierBoundMoreThanOnceInParameterList(identifier) => self
                .dcx()
                .create_err(errs::IdentifierBoundMoreThanOnceInParameterList { span, identifier }),
            ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(identifier) => self
                .dcx()
                .create_err(errs::IdentifierBoundMoreThanOnceInSamePattern { span, identifier }),
            ResolutionError::UndeclaredLabel { name, suggestion } => {
                let ((sub_reachable, sub_reachable_suggestion), sub_unreachable) = match suggestion
                {
                    // A reachable label with a similar name exists.
                    Some((ident, true)) => (
                        (
                            Some(errs::LabelWithSimilarNameReachable(ident.span)),
                            Some(errs::TryUsingSimilarlyNamedLabel {
                                span,
                                ident_name: ident.name,
                            }),
                        ),
                        None,
                    ),
                    // An unreachable label with a similar name exists.
                    Some((ident, false)) => (
                        (None, None),
                        Some(errs::UnreachableLabelWithSimilarNameExists {
                            ident_span: ident.span,
                        }),
                    ),
                    // No similarly-named labels exist.
                    None => ((None, None), None),
                };
                self.dcx().create_err(errs::UndeclaredLabel {
                    span,
                    name,
                    sub_reachable,
                    sub_reachable_suggestion,
                    sub_unreachable,
                })
            }
            ResolutionError::SelfImportsOnlyAllowedWithin { root, span_with_rename } => {
                // None of the suggestions below would help with a case like `use self`.
                let (suggestion, mpart_suggestion) = if root {
                    (None, None)
                } else {
                    // use foo::bar::self        -> foo::bar
                    // use foo::bar::self as abc -> foo::bar as abc
                    let suggestion = errs::SelfImportsOnlyAllowedWithinSuggestion { span };

                    // use foo::bar::self        -> foo::bar::{self}
                    // use foo::bar::self as abc -> foo::bar::{self as abc}
                    let mpart_suggestion = errs::SelfImportsOnlyAllowedWithinMultipartSuggestion {
                        multipart_start: span_with_rename.shrink_to_lo(),
                        multipart_end: span_with_rename.shrink_to_hi(),
                    };
                    (Some(suggestion), Some(mpart_suggestion))
                };
                self.dcx().create_err(errs::SelfImportsOnlyAllowedWithin {
                    span,
                    suggestion,
                    mpart_suggestion,
                })
            }
            ResolutionError::SelfImportCanOnlyAppearOnceInTheList => {
                self.dcx().create_err(errs::SelfImportCanOnlyAppearOnceInTheList { span })
            }
            ResolutionError::SelfImportOnlyInImportListWithNonEmptyPrefix => {
                self.dcx().create_err(errs::SelfImportOnlyInImportListWithNonEmptyPrefix { span })
            }
            ResolutionError::FailedToResolve { segment, label, suggestion, module } => {
                let mut err =
                    struct_span_code_err!(self.dcx(), span, E0433, "failed to resolve: {label}");
                err.span_label(span, label);

                if let Some((suggestions, msg, applicability)) = suggestion {
                    if suggestions.is_empty() {
                        err.help(msg);
                        return err;
                    }
                    err.multipart_suggestion(msg, suggestions, applicability);
                }
                if let Some(ModuleOrUniformRoot::Module(module)) = module
                    && let Some(module) = module.opt_def_id()
                    && let Some(segment) = segment
                {
                    self.find_cfg_stripped(&mut err, &segment, module);
                }

                err
            }
            ResolutionError::CannotCaptureDynamicEnvironmentInFnItem => {
                self.dcx().create_err(errs::CannotCaptureDynamicEnvironmentInFnItem { span })
            }
            ResolutionError::AttemptToUseNonConstantValueInConstant {
                ident,
                suggestion,
                current,
                type_span,
            } => {
                // let foo =...
                //     ^^^ given this Span
                // ------- get this Span to have an applicable suggestion

                // edit:
                // only do this if the const and usage of the non-constant value are on the same line
                // the further the two are apart, the higher the chance of the suggestion being wrong

                let sp = self
                    .tcx
                    .sess
                    .source_map()
                    .span_extend_to_prev_str(ident.span, current, true, false);

                let ((with, with_label), without) = match sp {
                    Some(sp) if !self.tcx.sess.source_map().is_multiline(sp) => {
                        let sp = sp
                            .with_lo(BytePos(sp.lo().0 - (current.len() as u32)))
                            .until(ident.span);
                        (
                        (Some(errs::AttemptToUseNonConstantValueInConstantWithSuggestion {
                                span: sp,
                                suggestion,
                                current,
                                type_span,
                            }), Some(errs::AttemptToUseNonConstantValueInConstantLabelWithSuggestion {span})),
                            None,
                        )
                    }
                    _ => (
                        (None, None),
                        Some(errs::AttemptToUseNonConstantValueInConstantWithoutSuggestion {
                            ident_span: ident.span,
                            suggestion,
                        }),
                    ),
                };

                self.dcx().create_err(errs::AttemptToUseNonConstantValueInConstant {
                    span,
                    with,
                    with_label,
                    without,
                })
            }
            ResolutionError::BindingShadowsSomethingUnacceptable {
                shadowing_binding,
                name,
                participle,
                article,
                shadowed_binding,
                shadowed_binding_span,
            } => self.dcx().create_err(errs::BindingShadowsSomethingUnacceptable {
                span,
                shadowing_binding,
                shadowed_binding,
                article,
                sub_suggestion: match (shadowing_binding, shadowed_binding) {
                    (
                        PatternSource::Match,
                        Res::Def(DefKind::Ctor(CtorOf::Variant | CtorOf::Struct, CtorKind::Fn), _),
                    ) => Some(errs::BindingShadowsSomethingUnacceptableSuggestion { span, name }),
                    _ => None,
                },
                shadowed_binding_span,
                participle,
                name,
            }),
            ResolutionError::ForwardDeclaredGenericParam(param, reason) => match reason {
                ForwardGenericParamBanReason::Default => {
                    self.dcx().create_err(errs::ForwardDeclaredGenericParam { param, span })
                }
                ForwardGenericParamBanReason::ConstParamTy => self
                    .dcx()
                    .create_err(errs::ForwardDeclaredGenericInConstParamTy { param, span }),
            },
            ResolutionError::ParamInTyOfConstParam { name } => {
                self.dcx().create_err(errs::ParamInTyOfConstParam { span, name })
            }
            ResolutionError::ParamInNonTrivialAnonConst { name, param_kind: is_type } => {
                self.dcx().create_err(errs::ParamInNonTrivialAnonConst {
                    span,
                    name,
                    param_kind: is_type,
                    help: self
                        .tcx
                        .sess
                        .is_nightly_build()
                        .then_some(errs::ParamInNonTrivialAnonConstHelp),
                })
            }
            ResolutionError::ParamInEnumDiscriminant { name, param_kind: is_type } => self
                .dcx()
                .create_err(errs::ParamInEnumDiscriminant { span, name, param_kind: is_type }),
            ResolutionError::ForwardDeclaredSelf(reason) => match reason {
                ForwardGenericParamBanReason::Default => {
                    self.dcx().create_err(errs::SelfInGenericParamDefault { span })
                }
                ForwardGenericParamBanReason::ConstParamTy => {
                    self.dcx().create_err(errs::SelfInConstGenericTy { span })
                }
            },
            ResolutionError::UnreachableLabel { name, definition_span, suggestion } => {
                let ((sub_suggestion_label, sub_suggestion), sub_unreachable_label) =
                    match suggestion {
                        // A reachable label with a similar name exists.
                        Some((ident, true)) => (
                            (
                                Some(errs::UnreachableLabelSubLabel { ident_span: ident.span }),
                                Some(errs::UnreachableLabelSubSuggestion {
                                    span,
                                    // intentionally taking 'ident.name' instead of 'ident' itself, as this
                                    // could be used in suggestion context
                                    ident_name: ident.name,
                                }),
                            ),
                            None,
                        ),
                        // An unreachable label with a similar name exists.
                        Some((ident, false)) => (
                            (None, None),
                            Some(errs::UnreachableLabelSubLabelUnreachable {
                                ident_span: ident.span,
                            }),
                        ),
                        // No similarly-named labels exist.
                        None => ((None, None), None),
                    };
                self.dcx().create_err(errs::UnreachableLabel {
                    span,
                    name,
                    definition_span,
                    sub_suggestion,
                    sub_suggestion_label,
                    sub_unreachable_label,
                })
            }
            ResolutionError::TraitImplMismatch {
                name,
                kind,
                code,
                trait_item_span,
                trait_path,
            } => self
                .dcx()
                .create_err(errors::TraitImplMismatch {
                    span,
                    name,
                    kind,
                    trait_path,
                    trait_item_span,
                })
                .with_code(code),
            ResolutionError::TraitImplDuplicate { name, trait_item_span, old_span } => self
                .dcx()
                .create_err(errs::TraitImplDuplicate { span, name, trait_item_span, old_span }),
            ResolutionError::InvalidAsmSym => self.dcx().create_err(errs::InvalidAsmSym { span }),
            ResolutionError::LowercaseSelf => self.dcx().create_err(errs::LowercaseSelf { span }),
            ResolutionError::BindingInNeverPattern => {
                self.dcx().create_err(errs::BindingInNeverPattern { span })
            }
        }
    }

    pub(crate) fn report_vis_error(
        &mut self,
        vis_resolution_error: VisResolutionError<'_>,
    ) -> ErrorGuaranteed {
        match vis_resolution_error {
            VisResolutionError::Relative2018(span, path) => {
                self.dcx().create_err(errs::Relative2018 {
                    span,
                    path_span: path.span,
                    // intentionally converting to String, as the text would also be used as
                    // in suggestion context
                    path_str: pprust::path_to_string(path),
                })
            }
            VisResolutionError::AncestorOnly(span) => {
                self.dcx().create_err(errs::AncestorOnly(span))
            }
            VisResolutionError::FailedToResolve(span, label, suggestion) => self.into_struct_error(
                span,
                ResolutionError::FailedToResolve { segment: None, label, suggestion, module: None },
            ),
            VisResolutionError::ExpectedFound(span, path_str, res) => {
                self.dcx().create_err(errs::ExpectedModuleFound { span, res, path_str })
            }
            VisResolutionError::Indeterminate(span) => {
                self.dcx().create_err(errs::Indeterminate(span))
            }
            VisResolutionError::ModuleOnly(span) => self.dcx().create_err(errs::ModuleOnly(span)),
        }
        .emit()
    }

    /// Lookup typo candidate in scope for a macro or import.
    fn early_lookup_typo_candidate(
        &mut self,
        scope_set: ScopeSet<'ra>,
        parent_scope: &ParentScope<'ra>,
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
                                .map(|(ident, _)| TypoSuggestion::typo_from_ident(*ident, res)),
                        );
                    }
                }
                Scope::DeriveHelpersCompat => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat);
                    if filter_fn(res) {
                        for derive in parent_scope.derives {
                            let parent_scope = &ParentScope { derives: &[], ..*parent_scope };
                            let Ok((Some(ext), _)) = this.resolve_macro_path(
                                derive,
                                Some(MacroKind::Derive),
                                parent_scope,
                                false,
                                false,
                                None,
                            ) else {
                                continue;
                            };
                            suggestions.extend(
                                ext.helper_attrs
                                    .iter()
                                    .map(|name| TypoSuggestion::typo_from_name(*name, res)),
                            );
                        }
                    }
                }
                Scope::MacroRules(macro_rules_scope) => {
                    if let MacroRulesScope::Binding(macro_rules_binding) = macro_rules_scope.get() {
                        let res = macro_rules_binding.binding.res();
                        if filter_fn(res) {
                            suggestions.push(TypoSuggestion::typo_from_ident(
                                macro_rules_binding.ident,
                                res,
                            ))
                        }
                    }
                }
                Scope::CrateRoot => {
                    let root_ident = Ident::new(kw::PathRoot, ident.span);
                    let root_module = this.resolve_crate_root(root_ident);
                    this.add_module_candidates(root_module, &mut suggestions, filter_fn, None);
                }
                Scope::Module(module, _) => {
                    this.add_module_candidates(module, &mut suggestions, filter_fn, None);
                }
                Scope::MacroUsePrelude => {
                    suggestions.extend(this.macro_use_prelude.iter().filter_map(
                        |(name, binding)| {
                            let res = binding.res();
                            filter_fn(res).then_some(TypoSuggestion::typo_from_name(*name, res))
                        },
                    ));
                }
                Scope::BuiltinAttrs => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Builtin(sym::dummy));
                    if filter_fn(res) {
                        suggestions.extend(
                            BUILTIN_ATTRIBUTES
                                .iter()
                                .map(|attr| TypoSuggestion::typo_from_name(attr.name, res)),
                        );
                    }
                }
                Scope::ExternPrelude => {
                    suggestions.extend(this.extern_prelude.iter().filter_map(|(ident, _)| {
                        let res = Res::Def(DefKind::Mod, CRATE_DEF_ID.to_def_id());
                        filter_fn(res).then_some(TypoSuggestion::typo_from_ident(*ident, res))
                    }));
                }
                Scope::ToolPrelude => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                    suggestions.extend(
                        this.registered_tools
                            .iter()
                            .map(|ident| TypoSuggestion::typo_from_ident(*ident, res)),
                    );
                }
                Scope::StdLibPrelude => {
                    if let Some(prelude) = this.prelude {
                        let mut tmp_suggestions = Vec::new();
                        this.add_module_candidates(prelude, &mut tmp_suggestions, filter_fn, None);
                        suggestions.extend(
                            tmp_suggestions
                                .into_iter()
                                .filter(|s| use_prelude.into() || this.is_builtin_macro(s.res)),
                        );
                    }
                }
                Scope::BuiltinTypes => {
                    suggestions.extend(PrimTy::ALL.iter().filter_map(|prim_ty| {
                        let res = Res::PrimTy(*prim_ty);
                        filter_fn(res)
                            .then_some(TypoSuggestion::typo_from_name(prim_ty.name(), res))
                    }))
                }
            }

            None::<()>
        });

        // Make sure error reporting is deterministic.
        suggestions.sort_by(|a, b| a.candidate.as_str().cmp(b.candidate.as_str()));

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
        parent_scope: &ParentScope<'ra>,
        start_module: Module<'ra>,
        crate_path: ThinVec<ast::PathSegment>,
        filter_fn: FilterFn,
    ) -> Vec<ImportSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        let mut candidates = Vec::new();
        let mut seen_modules = FxHashSet::default();
        let start_did = start_module.def_id();
        let mut worklist = vec![(
            start_module,
            ThinVec::<ast::PathSegment>::new(),
            true,
            start_did.is_local() || !self.tcx.is_doc_hidden(start_did),
            true,
        )];
        let mut worklist_via_import = vec![];

        while let Some((in_module, path_segments, accessible, doc_visible, is_stable)) =
            match worklist.pop() {
                None => worklist_via_import.pop(),
                Some(x) => Some(x),
            }
        {
            let in_module_is_extern = !in_module.def_id().is_local();
            in_module.for_each_child(self, |this, ident, ns, name_binding| {
                // Avoid non-importable candidates.
                if name_binding.is_assoc_item()
                    && !this.tcx.features().import_trait_associated_functions()
                {
                    return;
                }

                if ident.name == kw::Underscore {
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

                // #90113: Do not count an inaccessible reexported item as a candidate.
                if let NameBindingKind::Import { binding, .. } = name_binding.kind
                    && this.is_accessible_from(binding.vis, parent_scope.module)
                    && !this.is_accessible_from(name_binding.vis, parent_scope.module)
                {
                    return;
                }

                let res = name_binding.res();
                let did = match res {
                    Res::Def(DefKind::Ctor(..), did) => this.tcx.opt_parent(did),
                    _ => res.opt_def_id(),
                };
                let child_doc_visible = doc_visible
                    && did.is_none_or(|did| did.is_local() || !this.tcx.is_doc_hidden(did));

                // collect results based on the filter function
                // avoid suggesting anything from the same module in which we are resolving
                // avoid suggesting anything with a hygienic name
                if ident.name == lookup_ident.name
                    && ns == namespace
                    && in_module != parent_scope.module
                    && !ident.span.normalize_to_macros_2_0().from_expansion()
                    && filter_fn(res)
                {
                    // create the path
                    let mut segms = if lookup_ident.span.at_least_rust_2018() {
                        // crate-local absolute paths start with `crate::` in edition 2018
                        // FIXME: may also be stabilized for Rust 2015 (Issues #45477, #44660)
                        crate_path.clone()
                    } else {
                        ThinVec::new()
                    };
                    segms.append(&mut path_segments.clone());

                    segms.push(ast::PathSegment::from_ident(ident));
                    let path = Path { span: name_binding.span, segments: segms, tokens: None };

                    if child_accessible
                        // Remove invisible match if exists
                        && let Some(idx) = candidates
                            .iter()
                            .position(|v: &ImportSuggestion| v.did == did && !v.accessible)
                    {
                        candidates.remove(idx);
                    }

                    let is_stable = if is_stable
                        && let Some(did) = did
                        && this.is_stable(did, path.span)
                    {
                        true
                    } else {
                        false
                    };

                    // Rreplace unstable suggestions if we meet a new stable one,
                    // and do nothing if any other situation. For example, if we
                    // meet `std::ops::Range` after `std::range::legacy::Range`,
                    // we will remove the latter and then insert the former.
                    if is_stable
                        && let Some(idx) = candidates
                            .iter()
                            .position(|v: &ImportSuggestion| v.did == did && !v.is_stable)
                    {
                        candidates.remove(idx);
                    }

                    if candidates.iter().all(|v: &ImportSuggestion| v.did != did) {
                        // See if we're recommending TryFrom, TryInto, or FromIterator and add
                        // a note about editions
                        let note = if let Some(did) = did {
                            let requires_note = !did.is_local()
                                && this.tcx.get_attrs(did, sym::rustc_diagnostic_item).any(
                                    |attr| {
                                        [sym::TryInto, sym::TryFrom, sym::FromIterator]
                                            .map(|x| Some(x))
                                            .contains(&attr.value_str())
                                    },
                                );

                            requires_note.then(|| {
                                format!(
                                    "'{}' is included in the prelude starting in Edition 2021",
                                    path_names_to_string(&path)
                                )
                            })
                        } else {
                            None
                        };

                        candidates.push(ImportSuggestion {
                            did,
                            descr: res.descr(),
                            path,
                            accessible: child_accessible,
                            doc_visible: child_doc_visible,
                            note,
                            via_import,
                            is_stable,
                        });
                    }
                }

                // collect submodules to explore
                if let Some(module) = name_binding.module() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident));

                    let alias_import = if let NameBindingKind::Import { import, .. } =
                        name_binding.kind
                        && let ImportKind::ExternCrate { source: Some(_), .. } = import.kind
                        && import.parent_scope.expansion == parent_scope.expansion
                    {
                        true
                    } else {
                        false
                    };

                    let is_extern_crate_that_also_appears_in_prelude =
                        name_binding.is_extern_crate() && lookup_ident.span.at_least_rust_2018();

                    if !is_extern_crate_that_also_appears_in_prelude || alias_import {
                        // add the module to the lookup
                        if seen_modules.insert(module.def_id()) {
                            if via_import { &mut worklist_via_import } else { &mut worklist }.push(
                                (
                                    module,
                                    path_segments,
                                    child_accessible,
                                    child_doc_visible,
                                    is_stable && this.is_stable(module.def_id(), name_binding.span),
                                ),
                            );
                        }
                    }
                }
            })
        }

        candidates
    }

    fn is_stable(&self, did: DefId, span: Span) -> bool {
        if did.is_local() {
            return true;
        }

        match self.tcx.lookup_stability(did) {
            Some(Stability {
                level: attr::StabilityLevel::Unstable { implied_by, .. },
                feature,
                ..
            }) => {
                if span.allows_unstable(feature) {
                    true
                } else if self.tcx.features().enabled(feature) {
                    true
                } else if let Some(implied_by) = implied_by
                    && self.tcx.features().enabled(implied_by)
                {
                    true
                } else {
                    false
                }
            }
            Some(_) => true,
            None => false,
        }
    }

    /// When name resolution fails, this method can be used to look up candidate
    /// entities with the expected name. It allows filtering them using the
    /// supplied predicate (which should be used to only accept the types of
    /// definitions expected, e.g., traits). The lookup spans across all crates.
    ///
    /// N.B., the method does not look into imports, but this is not a problem,
    /// since we report the definitions (thus, the de-aliased imports).
    pub(crate) fn lookup_import_candidates<FilterFn>(
        &mut self,
        lookup_ident: Ident,
        namespace: Namespace,
        parent_scope: &ParentScope<'ra>,
        filter_fn: FilterFn,
    ) -> Vec<ImportSuggestion>
    where
        FilterFn: Fn(Res) -> bool,
    {
        let crate_path = thin_vec![ast::PathSegment::from_ident(Ident::with_dummy_span(kw::Crate))];
        let mut suggestions = self.lookup_import_candidates_from_module(
            lookup_ident,
            namespace,
            parent_scope,
            self.graph_root,
            crate_path,
            &filter_fn,
        );

        if lookup_ident.span.at_least_rust_2018() {
            for ident in self.extern_prelude.clone().into_keys() {
                if ident.span.from_expansion() {
                    // Idents are adjusted to the root context before being
                    // resolved in the extern prelude, so reporting this to the
                    // user is no help. This skips the injected
                    // `extern crate std` in the 2018 edition, which would
                    // otherwise cause duplicate suggestions.
                    continue;
                }
                let Some(crate_id) = self.crate_loader(|c| c.maybe_process_path_extern(ident.name))
                else {
                    continue;
                };

                let crate_def_id = crate_id.as_def_id();
                let crate_root = self.expect_module(crate_def_id);

                // Check if there's already an item in scope with the same name as the crate.
                // If so, we have to disambiguate the potential import suggestions by making
                // the paths *global* (i.e., by prefixing them with `::`).
                let needs_disambiguation =
                    self.resolutions(parent_scope.module).borrow().iter().any(
                        |(key, name_resolution)| {
                            if key.ns == TypeNS
                                && key.ident == ident
                                && let Some(binding) = name_resolution.borrow().binding
                            {
                                match binding.res() {
                                    // No disambiguation needed if the identically named item we
                                    // found in scope actually refers to the crate in question.
                                    Res::Def(_, def_id) => def_id != crate_def_id,
                                    Res::PrimTy(_) => true,
                                    _ => false,
                                }
                            } else {
                                false
                            }
                        },
                    );
                let mut crate_path = ThinVec::new();
                if needs_disambiguation {
                    crate_path.push(ast::PathSegment::path_root(rustc_span::DUMMY_SP));
                }
                crate_path.push(ast::PathSegment::from_ident(ident));

                suggestions.extend(self.lookup_import_candidates_from_module(
                    lookup_ident,
                    namespace,
                    parent_scope,
                    crate_root,
                    crate_path,
                    &filter_fn,
                ));
            }
        }

        suggestions
    }

    pub(crate) fn unresolved_macro_suggestions(
        &mut self,
        err: &mut Diag<'_>,
        macro_kind: MacroKind,
        parent_scope: &ParentScope<'ra>,
        ident: Ident,
        krate: &Crate,
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
        let (span, found_use) = match parent_scope.module.nearest_parent_mod().as_local() {
            Some(def_id) => UsePlacementFinder::check(krate, self.def_id_to_node_id(def_id)),
            None => (None, FoundUse::No),
        };
        show_candidates(
            self.tcx,
            err,
            span,
            &import_suggestions,
            Instead::No,
            found_use,
            DiagMode::Normal,
            vec![],
            "",
        );

        if macro_kind == MacroKind::Bang && ident.name == sym::macro_rules {
            let label_span = ident.span.shrink_to_hi();
            let mut spans = MultiSpan::from_span(label_span);
            spans.push_span_label(label_span, "put a macro name here");
            err.subdiagnostic(MaybeMissingMacroRulesName { spans });
            return;
        }

        if macro_kind == MacroKind::Derive && (ident.name == sym::Send || ident.name == sym::Sync) {
            err.subdiagnostic(ExplicitUnsafeTraits { span: ident.span, ident });
            return;
        }

        let unused_macro = self.unused_macros.iter().find_map(|(def_id, (_, unused_ident))| {
            if unused_ident.name == ident.name { Some((def_id, unused_ident)) } else { None }
        });

        if let Some((def_id, unused_ident)) = unused_macro {
            let scope = self.local_macro_def_scopes[&def_id];
            let parent_nearest = parent_scope.module.nearest_parent_mod();
            if Some(parent_nearest) == scope.opt_def_id() {
                match macro_kind {
                    MacroKind::Bang => {
                        err.subdiagnostic(MacroDefinedLater { span: unused_ident.span });
                        err.subdiagnostic(MacroSuggMovePosition { span: ident.span, ident });
                    }
                    MacroKind::Attr => {
                        err.subdiagnostic(MacroRulesNot::Attr { span: unused_ident.span, ident });
                    }
                    MacroKind::Derive => {
                        err.subdiagnostic(MacroRulesNot::Derive { span: unused_ident.span, ident });
                    }
                }

                return;
            }
        }

        if self.macro_names.contains(&ident.normalize_to_macros_2_0()) {
            err.subdiagnostic(AddedMacroUse);
            return;
        }

        if ident.name == kw::Default
            && let ModuleKind::Def(DefKind::Enum, def_id, _) = parent_scope.module.kind
        {
            let span = self.def_span(def_id);
            let source_map = self.tcx.sess.source_map();
            let head_span = source_map.guess_head_span(span);
            err.subdiagnostic(ConsiderAddingADerive {
                span: head_span.shrink_to_lo(),
                suggestion: "#[derive(Default)]\n".to_string(),
            });
        }
        for ns in [Namespace::MacroNS, Namespace::TypeNS, Namespace::ValueNS] {
            let Ok(binding) = self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::All(ns),
                parent_scope,
                None,
                false,
                None,
                None,
            ) else {
                continue;
            };

            let desc = match binding.res() {
                Res::Def(DefKind::Macro(MacroKind::Bang), _) => "a function-like macro".to_string(),
                Res::Def(DefKind::Macro(MacroKind::Attr), _) | Res::NonMacroAttr(..) => {
                    format!("an attribute: `#[{ident}]`")
                }
                Res::Def(DefKind::Macro(MacroKind::Derive), _) => {
                    format!("a derive macro: `#[derive({ident})]`")
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
            if let crate::NameBindingKind::Import { import, .. } = binding.kind
                && !import.span.is_dummy()
            {
                let note = errors::IdentImporterHereButItIsDesc {
                    span: import.span,
                    imported_ident: ident,
                    imported_ident_desc: &desc,
                };
                err.subdiagnostic(note);
                // Silence the 'unused import' warning we might get,
                // since this diagnostic already covers that import.
                self.record_use(ident, binding, Used::Other);
                return;
            }
            let note = errors::IdentInScopeButItIsDesc {
                imported_ident: ident,
                imported_ident_desc: &desc,
            };
            err.subdiagnostic(note);
            return;
        }
    }

    pub(crate) fn add_typo_suggestion(
        &self,
        err: &mut Diag<'_>,
        suggestion: Option<TypoSuggestion>,
        span: Span,
    ) -> bool {
        let suggestion = match suggestion {
            None => return false,
            // We shouldn't suggest underscore.
            Some(suggestion) if suggestion.candidate == kw::Underscore => return false,
            Some(suggestion) => suggestion,
        };

        let mut did_label_def_span = false;

        if let Some(def_span) = suggestion.res.opt_def_id().map(|def_id| self.def_span(def_id)) {
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
            let span = self.tcx.sess.source_map().guess_head_span(def_span);
            let candidate_descr = suggestion.res.descr();
            let candidate = suggestion.candidate;
            let label = match suggestion.target {
                SuggestionTarget::SimilarlyNamed => {
                    errors::DefinedHere::SimilarlyNamed { span, candidate_descr, candidate }
                }
                SuggestionTarget::SingleItem => {
                    errors::DefinedHere::SingleItem { span, candidate_descr, candidate }
                }
            };
            did_label_def_span = true;
            err.subdiagnostic(label);
        }

        let (span, msg, sugg) = if let SuggestionTarget::SimilarlyNamed = suggestion.target
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span)
            && let Some(span) = suggestion.span
            && let Some(candidate) = suggestion.candidate.as_str().strip_prefix('_')
            && snippet == candidate
        {
            let candidate = suggestion.candidate;
            // When the suggested binding change would be from `x` to `_x`, suggest changing the
            // original binding definition instead. (#60164)
            let msg = format!(
                "the leading underscore in `{candidate}` marks it as unused, consider renaming it to `{snippet}`"
            );
            if !did_label_def_span {
                err.span_label(span, format!("`{candidate}` defined here"));
            }
            (span, msg, snippet)
        } else {
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
            (span, msg, suggestion.candidate.to_ident_string())
        };
        err.span_suggestion(span, msg, sugg, Applicability::MaybeIncorrect);
        true
    }

    fn binding_description(&self, b: NameBinding<'_>, ident: Ident, from_prelude: bool) -> String {
        let res = b.res();
        if b.span.is_dummy() || !self.tcx.sess.source_map().is_span_accessible(b.span) {
            // These already contain the "built-in" prefix or look bad with it.
            let add_built_in =
                !matches!(b.res(), Res::NonMacroAttr(..) | Res::PrimTy(..) | Res::ToolMod);
            let (built_in, from) = if from_prelude {
                ("", " from prelude")
            } else if b.is_extern_crate()
                && !b.is_import()
                && self.tcx.sess.opts.externs.get(ident.as_str()).is_some()
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
            let introduced = if b.is_import_user_facing() { "imported" } else { "defined" };
            format!("the {thing} {introduced} here", thing = res.descr())
        }
    }

    fn ambiguity_diagnostics(&self, ambiguity_error: &AmbiguityError<'_>) -> AmbiguityErrorDiag {
        let AmbiguityError { kind, ident, b1, b2, misc1, misc2, .. } = *ambiguity_error;
        let (b1, b2, misc1, misc2, swapped) = if b2.span.is_dummy() && !b1.span.is_dummy() {
            // We have to print the span-less alternative first, otherwise formatting looks bad.
            (b2, b1, misc2, misc1, true)
        } else {
            (b1, b2, misc1, misc2, false)
        };
        let could_refer_to = |b: NameBinding<'_>, misc: AmbiguityErrorMisc, also: &str| {
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
            if b.is_extern_crate() && ident.span.at_least_rust_2018() {
                help_msgs.push(format!("use `::{ident}` to refer to this {thing} unambiguously"))
            }
            match misc {
                AmbiguityErrorMisc::SuggestCrate => help_msgs
                    .push(format!("use `crate::{ident}` to refer to this {thing} unambiguously")),
                AmbiguityErrorMisc::SuggestSelf => help_msgs
                    .push(format!("use `self::{ident}` to refer to this {thing} unambiguously")),
                AmbiguityErrorMisc::FromPrelude | AmbiguityErrorMisc::None => {}
            }

            (
                b.span,
                note_msg,
                help_msgs
                    .iter()
                    .enumerate()
                    .map(|(i, help_msg)| {
                        let or = if i == 0 { "" } else { "or " };
                        format!("{or}{help_msg}")
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let (b1_span, b1_note_msg, b1_help_msgs) = could_refer_to(b1, misc1, "");
        let (b2_span, b2_note_msg, b2_help_msgs) = could_refer_to(b2, misc2, " also");

        AmbiguityErrorDiag {
            msg: format!("`{ident}` is ambiguous"),
            span: ident.span,
            label_span: ident.span,
            label_msg: "ambiguous name".to_string(),
            note_msg: format!("ambiguous because of {}", kind.descr()),
            b1_span,
            b1_note_msg,
            b1_help_msgs,
            b2_span,
            b2_note_msg,
            b2_help_msgs,
        }
    }

    /// If the binding refers to a tuple struct constructor with fields,
    /// returns the span of its fields.
    fn ctor_fields_span(&self, binding: NameBinding<'_>) -> Option<Span> {
        let NameBindingKind::Res(Res::Def(
            DefKind::Ctor(CtorOf::Struct, CtorKind::Fn),
            ctor_def_id,
        )) = binding.kind
        else {
            return None;
        };

        let def_id = self.tcx.parent(ctor_def_id);
        self.field_idents(def_id)?.iter().map(|&f| f.span).reduce(Span::to) // None for `struct Foo()`
    }

    fn report_privacy_error(&mut self, privacy_error: &PrivacyError<'ra>) {
        let PrivacyError { ident, binding, outermost_res, parent_scope, single_nested, dedup_span } =
            *privacy_error;

        let res = binding.res();
        let ctor_fields_span = self.ctor_fields_span(binding);
        let plain_descr = res.descr().to_string();
        let nonimport_descr =
            if ctor_fields_span.is_some() { plain_descr + " constructor" } else { plain_descr };
        let import_descr = nonimport_descr.clone() + " import";
        let get_descr =
            |b: NameBinding<'_>| if b.is_import() { &import_descr } else { &nonimport_descr };

        // Print the primary message.
        let ident_descr = get_descr(binding);
        let mut err =
            self.dcx().create_err(errors::IsPrivate { span: ident.span, ident_descr, ident });

        let mut not_publicly_reexported = false;
        if let Some((this_res, outer_ident)) = outermost_res {
            let import_suggestions = self.lookup_import_candidates(
                outer_ident,
                this_res.ns().unwrap_or(Namespace::TypeNS),
                &parent_scope,
                &|res: Res| res == this_res,
            );
            let point_to_def = !show_candidates(
                self.tcx,
                &mut err,
                Some(dedup_span.until(outer_ident.span.shrink_to_hi())),
                &import_suggestions,
                Instead::Yes,
                FoundUse::Yes,
                DiagMode::Import { append: single_nested, unresolved_import: false },
                vec![],
                "",
            );
            // If we suggest importing a public re-export, don't point at the definition.
            if point_to_def && ident.span != outer_ident.span {
                not_publicly_reexported = true;
                let label = errors::OuterIdentIsNotPubliclyReexported {
                    span: outer_ident.span,
                    outer_ident_descr: this_res.descr(),
                    outer_ident,
                };
                err.subdiagnostic(label);
            }
        }

        let mut non_exhaustive = None;
        // If an ADT is foreign and marked as `non_exhaustive`, then that's
        // probably why we have the privacy error.
        // Otherwise, point out if the struct has any private fields.
        if let Some(def_id) = res.opt_def_id()
            && !def_id.is_local()
            && let Some(attr) = self.tcx.get_attr(def_id, sym::non_exhaustive)
        {
            non_exhaustive = Some(attr.span());
        } else if let Some(span) = ctor_fields_span {
            let label = errors::ConstructorPrivateIfAnyFieldPrivate { span };
            err.subdiagnostic(label);
            if let Res::Def(_, d) = res
                && let Some(fields) = self.field_visibility_spans.get(&d)
            {
                let spans = fields.iter().map(|span| *span).collect();
                let sugg =
                    errors::ConsiderMakingTheFieldPublic { spans, number_of_fields: fields.len() };
                err.subdiagnostic(sugg);
            }
        }

        let mut sugg_paths = vec![];
        if let Some(mut def_id) = res.opt_def_id() {
            // We can't use `def_path_str` in resolve.
            let mut path = vec![def_id];
            while let Some(parent) = self.tcx.opt_parent(def_id) {
                def_id = parent;
                if !def_id.is_top_level_module() {
                    path.push(def_id);
                } else {
                    break;
                }
            }
            // We will only suggest importing directly if it is accessible through that path.
            let path_names: Option<Vec<String>> = path
                .iter()
                .rev()
                .map(|def_id| {
                    self.tcx.opt_item_name(*def_id).map(|n| {
                        if def_id.is_top_level_module() {
                            "crate".to_string()
                        } else {
                            n.to_string()
                        }
                    })
                })
                .collect();
            if let Some(def_id) = path.get(0)
                && let Some(path) = path_names
            {
                if let Some(def_id) = def_id.as_local() {
                    if self.effective_visibilities.is_directly_public(def_id) {
                        sugg_paths.push((path, false));
                    }
                } else if self.is_accessible_from(self.tcx.visibility(def_id), parent_scope.module)
                {
                    sugg_paths.push((path, false));
                }
            }
        }

        // Print the whole import chain to make it easier to see what happens.
        let first_binding = binding;
        let mut next_binding = Some(binding);
        let mut next_ident = ident;
        let mut path = vec![];
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
                    ImportKind::Glob { .. }
                    | ImportKind::MacroUse { .. }
                    | ImportKind::MacroExport => Some(binding),
                    ImportKind::ExternCrate { .. } => None,
                },
                _ => None,
            };

            match binding.kind {
                NameBindingKind::Import { import, .. } => {
                    for segment in import.module_path.iter().skip(1) {
                        path.push(segment.ident.to_string());
                    }
                    sugg_paths.push((
                        path.iter()
                            .cloned()
                            .chain(vec![ident.to_string()].into_iter())
                            .collect::<Vec<_>>(),
                        true, // re-export
                    ));
                }
                NameBindingKind::Res(_) | NameBindingKind::Module(_) => {}
            }
            let first = binding == first_binding;
            let def_span = self.tcx.sess.source_map().guess_head_span(binding.span);
            let mut note_span = MultiSpan::from_span(def_span);
            if !first && binding.vis.is_public() {
                let desc = match binding.kind {
                    NameBindingKind::Import { .. } => "re-export",
                    _ => "directly",
                };
                note_span.push_span_label(def_span, format!("you could import this {desc}"));
            }
            // Final step in the import chain, point out if the ADT is `non_exhaustive`
            // which is probably why this privacy violation occurred.
            if next_binding.is_none()
                && let Some(span) = non_exhaustive
            {
                note_span.push_span_label(
                    span,
                    "cannot be constructed because it is `#[non_exhaustive]`",
                );
            }
            let note = errors::NoteAndRefersToTheItemDefinedHere {
                span: note_span,
                binding_descr: get_descr(binding),
                binding_name: name,
                first,
                dots: next_binding.is_some(),
            };
            err.subdiagnostic(note);
        }
        // We prioritize shorter paths, non-core imports and direct imports over the alternatives.
        sugg_paths.sort_by_key(|(p, reexport)| (p.len(), p[0] == "core", *reexport));
        for (sugg, reexport) in sugg_paths {
            if not_publicly_reexported {
                break;
            }
            if sugg.len() <= 1 {
                // A single path segment suggestion is wrong. This happens on circular imports.
                // `tests/ui/imports/issue-55884-2.rs`
                continue;
            }
            let path = sugg.join("::");
            let sugg = if reexport {
                errors::ImportIdent::ThroughReExport { span: dedup_span, ident, path }
            } else {
                errors::ImportIdent::Directly { span: dedup_span, ident, path }
            };
            err.subdiagnostic(sugg);
            break;
        }

        err.emit();
    }

    pub(crate) fn find_similarly_named_module_or_crate(
        &mut self,
        ident: Symbol,
        current_module: Module<'ra>,
    ) -> Option<Symbol> {
        let mut candidates = self
            .extern_prelude
            .keys()
            .map(|ident| ident.name)
            .chain(
                self.module_map
                    .iter()
                    .filter(|(_, module)| {
                        current_module.is_ancestor_of(**module) && current_module != **module
                    })
                    .flat_map(|(_, module)| module.kind.name()),
            )
            .filter(|c| !c.to_string().is_empty())
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();
        find_best_match_for_name(&candidates, ident, None).filter(|sugg| *sugg != ident)
    }

    pub(crate) fn report_path_resolution_error(
        &mut self,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'ra>,
        ribs: Option<&PerNS<Vec<Rib<'ra>>>>,
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
        module: Option<ModuleOrUniformRoot<'ra>>,
        failed_segment_idx: usize,
        ident: Ident,
    ) -> (String, Option<Suggestion>) {
        let is_last = failed_segment_idx == path.len() - 1;
        let ns = if is_last { opt_ns.unwrap_or(TypeNS) } else { TypeNS };
        let module_res = match module {
            Some(ModuleOrUniformRoot::Module(module)) => module.res(),
            _ => None,
        };
        if module_res == self.graph_root.res() {
            let is_mod = |res| matches!(res, Res::Def(DefKind::Mod, _));
            let mut candidates = self.lookup_import_candidates(ident, TypeNS, parent_scope, is_mod);
            candidates
                .sort_by_cached_key(|c| (c.path.segments.len(), pprust::path_to_string(&c.path)));
            if let Some(candidate) = candidates.get(0) {
                let path = {
                    // remove the possible common prefix of the path
                    let len = candidate.path.segments.len();
                    let start_index = (0..=failed_segment_idx.min(len - 1))
                        .find(|&i| path[i].ident.name != candidate.path.segments[i].ident.name)
                        .unwrap_or_default();
                    let segments =
                        (start_index..len).map(|s| candidate.path.segments[s].clone()).collect();
                    Path { segments, span: Span::default(), tokens: None }
                };
                (
                    String::from("unresolved import"),
                    Some((
                        vec![(ident.span, pprust::path_to_string(&path))],
                        String::from("a similar path exists"),
                        Applicability::MaybeIncorrect,
                    )),
                )
            } else if ident.name == sym::core {
                (
                    format!("you might be missing crate `{ident}`"),
                    Some((
                        vec![(ident.span, "std".to_string())],
                        "try using `std` instead of `core`".to_string(),
                        Applicability::MaybeIncorrect,
                    )),
                )
            } else if ident.name == kw::Underscore {
                (format!("`_` is not a valid crate or module name"), None)
            } else if self.tcx.sess.is_rust_2015() {
                (
                    format!("use of unresolved module or unlinked crate `{ident}`"),
                    Some((
                        vec![(
                            self.current_crate_outer_attr_insert_span,
                            format!("extern crate {ident};\n"),
                        )],
                        if was_invoked_from_cargo() {
                            format!(
                                "if you wanted to use a crate named `{ident}`, use `cargo add {ident}` \
                             to add it to your `Cargo.toml` and import it in your code",
                            )
                        } else {
                            format!(
                                "you might be missing a crate named `{ident}`, add it to your \
                                 project and import it in your code",
                            )
                        },
                        Applicability::MaybeIncorrect,
                    )),
                )
            } else {
                (format!("could not find `{ident}` in the crate root"), None)
            }
        } else if failed_segment_idx > 0 {
            let parent = path[failed_segment_idx - 1].ident.name;
            let parent = match parent {
                // ::foo is mounted at the crate root for 2015, and is the extern
                // prelude for 2018+
                kw::PathRoot if self.tcx.sess.edition() > Edition::Edition2015 => {
                    "the list of imported crates".to_owned()
                }
                kw::PathRoot | kw::Crate => "the crate root".to_owned(),
                _ => format!("`{parent}`"),
            };

            let mut msg = format!("could not find `{ident}` in {parent}");
            if ns == TypeNS || ns == ValueNS {
                let ns_to_try = if ns == TypeNS { ValueNS } else { TypeNS };
                let binding = if let Some(module) = module {
                    self.resolve_ident_in_module(
                        module,
                        ident,
                        ns_to_try,
                        parent_scope,
                        None,
                        ignore_binding,
                        ignore_import,
                    )
                    .ok()
                } else if let Some(ribs) = ribs
                    && let Some(TypeNS | ValueNS) = opt_ns
                {
                    assert!(ignore_import.is_none());
                    match self.resolve_ident_in_lexical_scope(
                        ident,
                        ns_to_try,
                        parent_scope,
                        None,
                        &ribs[ns_to_try],
                        ignore_binding,
                    ) {
                        // we found a locally-imported or available item/module
                        Some(LexicalScopeBinding::Item(binding)) => Some(binding),
                        _ => None,
                    }
                } else {
                    self.early_resolve_ident_in_lexical_scope(
                        ident,
                        ScopeSet::All(ns_to_try),
                        parent_scope,
                        None,
                        false,
                        ignore_binding,
                        ignore_import,
                    )
                    .ok()
                };
                if let Some(binding) = binding {
                    let mut found = |what| {
                        msg = format!(
                            "expected {}, found {} `{}` in {}",
                            ns.descr(),
                            what,
                            ident,
                            parent
                        )
                    };
                    if binding.module().is_some() {
                        found("module")
                    } else {
                        match binding.res() {
                            // Avoid using TyCtxt::def_kind_descr in the resolver, because it
                            // indirectly *calls* the resolver, and would cause a query cycle.
                            Res::Def(kind, id) => found(kind.descr(id)),
                            _ => found(ns_to_try.descr()),
                        }
                    }
                };
            }
            (msg, None)
        } else if ident.name == kw::SelfUpper {
            // As mentioned above, `opt_ns` being `None` indicates a module path in import.
            // We can use this to improve a confusing error for, e.g. `use Self::Variant` in an
            // impl
            if opt_ns.is_none() {
                ("`Self` cannot be used in imports".to_string(), None)
            } else {
                (
                    "`Self` is only available in impls, traits, and type definitions".to_string(),
                    None,
                )
            }
        } else if ident.name.as_str().chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            // Check whether the name refers to an item in the value namespace.
            let binding = if let Some(ribs) = ribs {
                assert!(ignore_import.is_none());
                self.resolve_ident_in_lexical_scope(
                    ident,
                    ValueNS,
                    parent_scope,
                    None,
                    &ribs[ValueNS],
                    ignore_binding,
                )
            } else {
                None
            };
            let match_span = match binding {
                // Name matches a local variable. For example:
                // ```
                // fn f() {
                //     let Foo: &str = "";
                //     println!("{}", Foo::Bar); // Name refers to local
                //                               // variable `Foo`.
                // }
                // ```
                Some(LexicalScopeBinding::Res(Res::Local(id))) => {
                    Some(*self.pat_span_map.get(&id).unwrap())
                }
                // Name matches item from a local name binding
                // created by `use` declaration. For example:
                // ```
                // pub Foo: &str = "";
                //
                // mod submod {
                //     use super::Foo;
                //     println!("{}", Foo::Bar); // Name refers to local
                //                               // binding `Foo`.
                // }
                // ```
                Some(LexicalScopeBinding::Item(name_binding)) => Some(name_binding.span),
                _ => None,
            };
            let suggestion = match_span.map(|span| {
                (
                    vec![(span, String::from(""))],
                    format!("`{ident}` is defined here, but is not a type"),
                    Applicability::MaybeIncorrect,
                )
            });

            (format!("use of undeclared type `{ident}`"), suggestion)
        } else {
            let mut suggestion = None;
            if ident.name == sym::alloc {
                suggestion = Some((
                    vec![],
                    String::from("add `extern crate alloc` to use the `alloc` crate"),
                    Applicability::MaybeIncorrect,
                ))
            }

            suggestion = suggestion.or_else(|| {
                self.find_similarly_named_module_or_crate(ident.name, parent_scope.module).map(
                    |sugg| {
                        (
                            vec![(ident.span, sugg.to_string())],
                            String::from("there is a crate or module with a similar name"),
                            Applicability::MaybeIncorrect,
                        )
                    },
                )
            });
            if let Ok(binding) = self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::All(ValueNS),
                parent_scope,
                None,
                false,
                ignore_binding,
                ignore_import,
            ) {
                let descr = binding.res().descr();
                (format!("{descr} `{ident}` is not a crate or module"), suggestion)
            } else {
                let suggestion = if suggestion.is_some() {
                    suggestion
                } else if was_invoked_from_cargo() {
                    Some((
                        vec![],
                        format!(
                            "if you wanted to use a crate named `{ident}`, use `cargo add {ident}` \
                             to add it to your `Cargo.toml`",
                        ),
                        Applicability::MaybeIncorrect,
                    ))
                } else {
                    Some((
                        vec![],
                        format!("you might be missing a crate named `{ident}`",),
                        Applicability::MaybeIncorrect,
                    ))
                };
                (format!("use of unresolved module or unlinked crate `{ident}`"), suggestion)
            }
        }
    }

    /// Adds suggestions for a path that cannot be resolved.
    #[instrument(level = "debug", skip(self, parent_scope))]
    pub(crate) fn make_path_suggestion(
        &mut self,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'ra>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        match path[..] {
            // `{{root}}::ident::...` on both editions.
            // On 2015 `{{root}}` is usually added implicitly.
            [first, second, ..]
                if first.ident.name == kw::PathRoot && !second.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018.
            [first, ..]
                if first.ident.span.at_least_rust_2018()
                    && !first.ident.is_path_segment_keyword() =>
            {
                // Insert a placeholder that's later replaced by `self`/`super`/etc.
                path.insert(0, Segment::from_ident(Ident::dummy()));
            }
            _ => return None,
        }

        self.make_missing_self_suggestion(path.clone(), parent_scope)
            .or_else(|| self.make_missing_crate_suggestion(path.clone(), parent_scope))
            .or_else(|| self.make_missing_super_suggestion(path.clone(), parent_scope))
            .or_else(|| self.make_external_crate_suggestion(path, parent_scope))
    }

    /// Suggest a missing `self::` if that resolves to an correct module.
    ///
    /// ```text
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `self::foo`?
    /// ```
    #[instrument(level = "debug", skip(self, parent_scope))]
    fn make_missing_self_suggestion(
        &mut self,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'ra>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `self` and check if that is valid.
        path[0].ident.name = kw::SelfLower;
        let result = self.maybe_resolve_path(&path, None, parent_scope, None);
        debug!(?path, ?result);
        if let PathResult::Module(..) = result { Some((path, None)) } else { None }
    }

    /// Suggests a missing `crate::` if that resolves to an correct module.
    ///
    /// ```text
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `crate::foo`?
    /// ```
    #[instrument(level = "debug", skip(self, parent_scope))]
    fn make_missing_crate_suggestion(
        &mut self,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'ra>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Crate;
        let result = self.maybe_resolve_path(&path, None, parent_scope, None);
        debug!(?path, ?result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                Some(
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>"
                        .to_string(),
                ),
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
    #[instrument(level = "debug", skip(self, parent_scope))]
    fn make_missing_super_suggestion(
        &mut self,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'ra>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Super;
        let result = self.maybe_resolve_path(&path, None, parent_scope, None);
        debug!(?path, ?result);
        if let PathResult::Module(..) = result { Some((path, None)) } else { None }
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
    #[instrument(level = "debug", skip(self, parent_scope))]
    fn make_external_crate_suggestion(
        &mut self,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'ra>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        if path[1].ident.span.is_rust_2015() {
            return None;
        }

        // Sort extern crate names in *reverse* order to get
        // 1) some consistent ordering for emitted diagnostics, and
        // 2) `std` suggestions before `core` suggestions.
        let mut extern_crate_names =
            self.extern_prelude.keys().map(|ident| ident.name).collect::<Vec<_>>();
        extern_crate_names.sort_by(|a, b| b.as_str().cmp(a.as_str()));

        for name in extern_crate_names.into_iter() {
            // Replace first ident with a crate name and check if that is valid.
            path[0].ident.name = name;
            let result = self.maybe_resolve_path(&path, None, parent_scope, None);
            debug!(?path, ?name, ?result);
            if let PathResult::Module(..) = result {
                return Some((path, None));
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
        import: Import<'ra>,
        module: ModuleOrUniformRoot<'ra>,
        ident: Ident,
    ) -> Option<(Option<Suggestion>, Option<String>)> {
        let ModuleOrUniformRoot::Module(mut crate_module) = module else {
            return None;
        };

        while let Some(parent) = crate_module.parent {
            crate_module = parent;
        }

        if module == ModuleOrUniformRoot::Module(crate_module) {
            // Don't make a suggestion if the import was already from the root of the crate.
            return None;
        }

        let resolutions = self.resolutions(crate_module).borrow();
        let binding_key = BindingKey::new(ident, MacroNS);
        let resolution = resolutions.get(&binding_key)?;
        let binding = resolution.borrow().binding()?;
        let Res::Def(DefKind::Macro(MacroKind::Bang), _) = binding.res() else {
            return None;
        };
        let module_name = crate_module.kind.name().unwrap_or(kw::Crate);
        let import_snippet = match import.kind {
            ImportKind::Single { source, target, .. } if source != target => {
                format!("{source} as {target}")
            }
            _ => format!("{ident}"),
        };

        let mut corrections: Vec<(Span, String)> = Vec::new();
        if !import.is_nested() {
            // Assume this is the easy case of `use issue_59764::foo::makro;` and just remove
            // intermediate segments.
            corrections.push((import.span, format!("{module_name}::{import_snippet}")));
        } else {
            // Find the binding span (and any trailing commas and spaces).
            //   ie. `use a::b::{c, d, e};`
            //                      ^^^
            let (found_closing_brace, binding_span) = find_span_of_binding_until_next_binding(
                self.tcx.sess,
                import.span,
                import.use_span,
            );
            debug!(found_closing_brace, ?binding_span);

            let mut removal_span = binding_span;

            // If the binding span ended with a closing brace, as in the below example:
            //   ie. `use a::b::{c, d};`
            //                      ^
            // Then expand the span of characters to remove to include the previous
            // binding's trailing comma.
            //   ie. `use a::b::{c, d};`
            //                    ^^^
            if found_closing_brace
                && let Some(previous_span) =
                    extend_span_to_previous_binding(self.tcx.sess, binding_span)
            {
                debug!(?previous_span);
                removal_span = removal_span.with_lo(previous_span.lo());
            }
            debug!(?removal_span);

            // Remove the `removal_span`.
            corrections.push((removal_span, "".to_string()));

            // Find the span after the crate name and if it has nested imports immediately
            // after the crate name already.
            //   ie. `use a::b::{c, d};`
            //               ^^^^^^^^^
            //   or  `use a::{b, c, d}};`
            //               ^^^^^^^^^^^
            let (has_nested, after_crate_name) =
                find_span_immediately_after_crate_name(self.tcx.sess, import.use_span);
            debug!(has_nested, ?after_crate_name);

            let source_map = self.tcx.sess.source_map();

            // Make sure this is actually crate-relative.
            let is_definitely_crate = import
                .module_path
                .first()
                .is_some_and(|f| f.ident.name != kw::SelfLower && f.ident.name != kw::Super);

            // Add the import to the start, with a `{` if required.
            let start_point = source_map.start_point(after_crate_name);
            if is_definitely_crate
                && let Ok(start_snippet) = source_map.span_to_snippet(start_point)
            {
                corrections.push((
                    start_point,
                    if has_nested {
                        // In this case, `start_snippet` must equal '{'.
                        format!("{start_snippet}{import_snippet}, ")
                    } else {
                        // In this case, add a `{`, then the moved import, then whatever
                        // was there before.
                        format!("{{{import_snippet}, {start_snippet}")
                    },
                ));

                // Add a `};` to the end if nested, matching the `{` added at the start.
                if !has_nested {
                    corrections.push((source_map.end_point(after_crate_name), "};".to_string()));
                }
            } else {
                // If the root import is module-relative, add the import separately
                corrections.push((
                    import.use_span.shrink_to_lo(),
                    format!("use {module_name}::{import_snippet};\n"),
                ));
            }
        }

        let suggestion = Some((
            corrections,
            String::from("a macro with this name exists at the root of the crate"),
            Applicability::MaybeIncorrect,
        ));
        Some((
            suggestion,
            Some(
                "this could be because a macro annotated with `#[macro_export]` will be exported \
            at the root of the crate instead of the module where it is defined"
                    .to_string(),
            ),
        ))
    }

    /// Finds a cfg-ed out item inside `module` with the matching name.
    pub(crate) fn find_cfg_stripped(&self, err: &mut Diag<'_>, segment: &Symbol, module: DefId) {
        let local_items;
        let symbols = if module.is_local() {
            local_items = self
                .stripped_cfg_items
                .iter()
                .filter_map(|item| {
                    let parent_module = self.opt_local_def_id(item.parent_module)?.to_def_id();
                    Some(StrippedCfgItem {
                        parent_module,
                        ident: item.ident,
                        cfg: item.cfg.clone(),
                    })
                })
                .collect::<Vec<_>>();
            local_items.as_slice()
        } else {
            self.tcx.stripped_cfg_items(module.krate)
        };

        for &StrippedCfgItem { parent_module, ident, ref cfg } in symbols {
            if ident.name != *segment {
                continue;
            }

            fn comes_from_same_module_for_glob(
                r: &Resolver<'_, '_>,
                parent_module: DefId,
                module: DefId,
                visited: &mut FxHashMap<DefId, bool>,
            ) -> bool {
                if let Some(&cached) = visited.get(&parent_module) {
                    // this branch is prevent from being called recursively infinity,
                    // because there has some cycles in globs imports,
                    // see more spec case at `tests/ui/cfg/diagnostics-reexport-2.rs#reexport32`
                    return cached;
                }
                visited.insert(parent_module, false);
                let res = r.module_map.get(&parent_module).is_some_and(|m| {
                    for importer in m.glob_importers.borrow().iter() {
                        if let Some(next_parent_module) = importer.parent_scope.module.opt_def_id()
                        {
                            if next_parent_module == module
                                || comes_from_same_module_for_glob(
                                    r,
                                    next_parent_module,
                                    module,
                                    visited,
                                )
                            {
                                return true;
                            }
                        }
                    }
                    false
                });
                visited.insert(parent_module, res);
                res
            }

            let comes_from_same_module = parent_module == module
                || comes_from_same_module_for_glob(
                    self,
                    parent_module,
                    module,
                    &mut Default::default(),
                );
            if !comes_from_same_module {
                continue;
            }

            let note = errors::FoundItemConfigureOut { span: ident.span };
            err.subdiagnostic(note);

            if let MetaItemKind::List(nested) = &cfg.kind
                && let MetaItemInner::MetaItem(meta_item) = &nested[0]
                && let MetaItemKind::NameValue(feature_name) = &meta_item.kind
            {
                let note = errors::ItemWasBehindFeature {
                    feature: feature_name.symbol,
                    span: meta_item.span,
                };
                err.subdiagnostic(note);
            } else {
                let note = errors::ItemWasCfgOut { span: cfg.span };
                err.subdiagnostic(note);
            }
        }
    }
}

/// Given a `binding_span` of a binding within a use statement:
///
/// ```ignore (illustrative)
/// use foo::{a, b, c};
/// //           ^
/// ```
///
/// then return the span until the next binding or the end of the statement:
///
/// ```ignore (illustrative)
/// use foo::{a, b, c};
/// //           ^^^
/// ```
fn find_span_of_binding_until_next_binding(
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
/// ```ignore (illustrative)
/// use foo::a::{a, b, c};
/// //            ^^--- binding span
/// //            |
/// //            returned span
///
/// use foo::{a, b, c};
/// //        --- binding span
/// ```
fn extend_span_to_previous_binding(sess: &Session, binding_span: Span) -> Option<Span> {
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
/// ```ignore (illustrative)
/// use foo::a::{b, c};
/// //       ^^^^^^^^^^ -- false
///
/// use foo::{a, b, c};
/// //       ^^^^^^^^^^ -- true
///
/// use foo::{a, b::{c, d}};
/// //       ^^^^^^^^^^^^^^^ -- true
/// ```
#[instrument(level = "debug", skip(sess))]
fn find_span_immediately_after_crate_name(sess: &Session, use_span: Span) -> (bool, Span) {
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

/// A suggestion has already been emitted, change the wording slightly to clarify that both are
/// independent options.
enum Instead {
    Yes,
    No,
}

/// Whether an existing place with an `use` item was found.
enum FoundUse {
    Yes,
    No,
}

/// Whether a binding is part of a pattern or a use statement. Used for diagnostics.
pub(crate) enum DiagMode {
    Normal,
    /// The binding is part of a pattern
    Pattern,
    /// The binding is part of a use statement
    Import {
        /// `true` means diagnostics is for unresolved import
        unresolved_import: bool,
        /// `true` mean add the tips afterward for case `use a::{b,c}`,
        /// rather than replacing within.
        append: bool,
    },
}

pub(crate) fn import_candidates(
    tcx: TyCtxt<'_>,
    err: &mut Diag<'_>,
    // This is `None` if all placement locations are inside expansions
    use_placement_span: Option<Span>,
    candidates: &[ImportSuggestion],
    mode: DiagMode,
    append: &str,
) {
    show_candidates(
        tcx,
        err,
        use_placement_span,
        candidates,
        Instead::Yes,
        FoundUse::Yes,
        mode,
        vec![],
        append,
    );
}

type PathString<'a> = (String, &'a str, Option<Span>, &'a Option<String>, bool);

/// When an entity with a given name is not available in scope, we search for
/// entities with that name in all crates. This method allows outputting the
/// results of this search in a programmer-friendly way. If any entities are
/// found and suggested, returns `true`, otherwise returns `false`.
fn show_candidates(
    tcx: TyCtxt<'_>,
    err: &mut Diag<'_>,
    // This is `None` if all placement locations are inside expansions
    use_placement_span: Option<Span>,
    candidates: &[ImportSuggestion],
    instead: Instead,
    found_use: FoundUse,
    mode: DiagMode,
    path: Vec<Segment>,
    append: &str,
) -> bool {
    if candidates.is_empty() {
        return false;
    }

    let mut showed = false;
    let mut accessible_path_strings: Vec<PathString<'_>> = Vec::new();
    let mut inaccessible_path_strings: Vec<PathString<'_>> = Vec::new();

    candidates.iter().for_each(|c| {
        if c.accessible {
            // Don't suggest `#[doc(hidden)]` items from other crates
            if c.doc_visible {
                accessible_path_strings.push((
                    pprust::path_to_string(&c.path),
                    c.descr,
                    c.did.and_then(|did| Some(tcx.source_span(did.as_local()?))),
                    &c.note,
                    c.via_import,
                ))
            }
        } else {
            inaccessible_path_strings.push((
                pprust::path_to_string(&c.path),
                c.descr,
                c.did.and_then(|did| Some(tcx.source_span(did.as_local()?))),
                &c.note,
                c.via_import,
            ))
        }
    });

    // we want consistent results across executions, but candidates are produced
    // by iterating through a hash map, so make sure they are ordered:
    for path_strings in [&mut accessible_path_strings, &mut inaccessible_path_strings] {
        path_strings.sort_by(|a, b| a.0.cmp(&b.0));
        path_strings.dedup_by(|a, b| a.0 == b.0);
        let core_path_strings =
            path_strings.extract_if(.., |p| p.0.starts_with("core::")).collect::<Vec<_>>();
        let std_path_strings =
            path_strings.extract_if(.., |p| p.0.starts_with("std::")).collect::<Vec<_>>();
        let foreign_crate_path_strings =
            path_strings.extract_if(.., |p| !p.0.starts_with("crate::")).collect::<Vec<_>>();

        // We list the `crate` local paths first.
        // Then we list the `std`/`core` paths.
        if std_path_strings.len() == core_path_strings.len() {
            // Do not list `core::` paths if we are already listing the `std::` ones.
            path_strings.extend(std_path_strings);
        } else {
            path_strings.extend(std_path_strings);
            path_strings.extend(core_path_strings);
        }
        // List all paths from foreign crates last.
        path_strings.extend(foreign_crate_path_strings);
    }

    if !accessible_path_strings.is_empty() {
        let (determiner, kind, s, name, through) =
            if let [(name, descr, _, _, via_import)] = &accessible_path_strings[..] {
                (
                    "this",
                    *descr,
                    "",
                    format!(" `{name}`"),
                    if *via_import { " through its public re-export" } else { "" },
                )
            } else {
                // Get the unique item kinds and if there's only one, we use the right kind name
                // instead of the more generic "items".
                let kinds = accessible_path_strings
                    .iter()
                    .map(|(_, descr, _, _, _)| *descr)
                    .collect::<UnordSet<&str>>();
                let kind = if let Some(kind) = kinds.get_only() { kind } else { "item" };
                let s = if kind.ends_with('s') { "es" } else { "s" };

                ("one of these", kind, s, String::new(), "")
            };

        let instead = if let Instead::Yes = instead { " instead" } else { "" };
        let mut msg = if let DiagMode::Pattern = mode {
            format!(
                "if you meant to match on {kind}{s}{instead}{name}, use the full path in the \
                 pattern",
            )
        } else {
            format!("consider importing {determiner} {kind}{s}{through}{instead}")
        };

        for note in accessible_path_strings.iter().flat_map(|cand| cand.3.as_ref()) {
            err.note(note.clone());
        }

        let append_candidates = |msg: &mut String, accessible_path_strings: Vec<PathString<'_>>| {
            msg.push(':');

            for candidate in accessible_path_strings {
                msg.push('\n');
                msg.push_str(&candidate.0);
            }
        };

        if let Some(span) = use_placement_span {
            let (add_use, trailing) = match mode {
                DiagMode::Pattern => {
                    err.span_suggestions(
                        span,
                        msg,
                        accessible_path_strings.into_iter().map(|a| a.0),
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
                DiagMode::Import { .. } => ("", ""),
                DiagMode::Normal => ("use ", ";\n"),
            };
            for candidate in &mut accessible_path_strings {
                // produce an additional newline to separate the new use statement
                // from the directly following item.
                let additional_newline = if let FoundUse::No = found_use
                    && let DiagMode::Normal = mode
                {
                    "\n"
                } else {
                    ""
                };
                candidate.0 =
                    format!("{add_use}{}{append}{trailing}{additional_newline}", candidate.0);
            }

            match mode {
                DiagMode::Import { append: true, .. } => {
                    append_candidates(&mut msg, accessible_path_strings);
                    err.span_help(span, msg);
                }
                _ => {
                    err.span_suggestions_with_style(
                        span,
                        msg,
                        accessible_path_strings.into_iter().map(|a| a.0),
                        Applicability::MaybeIncorrect,
                        SuggestionStyle::ShowAlways,
                    );
                }
            }

            if let [first, .., last] = &path[..] {
                let sp = first.ident.span.until(last.ident.span);
                // Our suggestion is empty, so make sure the span is not empty (or we'd ICE).
                // Can happen for derive-generated spans.
                if sp.can_be_used_for_suggestions() && !sp.is_empty() {
                    err.span_suggestion_verbose(
                        sp,
                        format!("if you import `{}`, refer to it directly", last.ident),
                        "",
                        Applicability::Unspecified,
                    );
                }
            }
        } else {
            append_candidates(&mut msg, accessible_path_strings);
            err.help(msg);
        }
        showed = true;
    }
    if !inaccessible_path_strings.is_empty()
        && (!matches!(mode, DiagMode::Import { unresolved_import: false, .. }))
    {
        let prefix =
            if let DiagMode::Pattern = mode { "you might have meant to match on " } else { "" };
        if let [(name, descr, source_span, note, _)] = &inaccessible_path_strings[..] {
            let msg = format!(
                "{prefix}{descr} `{name}`{} exists but is inaccessible",
                if let DiagMode::Pattern = mode { ", which" } else { "" }
            );

            if let Some(source_span) = source_span {
                let span = tcx.sess.source_map().guess_head_span(*source_span);
                let mut multi_span = MultiSpan::from_span(span);
                multi_span.push_span_label(span, "not accessible");
                err.span_note(multi_span, msg);
            } else {
                err.note(msg);
            }
            if let Some(note) = (*note).as_deref() {
                err.note(note.to_string());
            }
        } else {
            let (_, descr_first, _, _, _) = &inaccessible_path_strings[0];
            let descr = if inaccessible_path_strings
                .iter()
                .skip(1)
                .all(|(_, descr, _, _, _)| descr == descr_first)
            {
                descr_first
            } else {
                "item"
            };
            let plural_descr =
                if descr.ends_with('s') { format!("{descr}es") } else { format!("{descr}s") };

            let mut msg = format!("{prefix}these {plural_descr} exist but are inaccessible");
            let mut has_colon = false;

            let mut spans = Vec::new();
            for (name, _, source_span, _, _) in &inaccessible_path_strings {
                if let Some(source_span) = source_span {
                    let span = tcx.sess.source_map().guess_head_span(*source_span);
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
                multi_span.push_span_label(span, format!("`{name}`: not accessible"));
            }

            for note in inaccessible_path_strings.iter().flat_map(|cand| cand.3.as_ref()) {
                err.note(note.clone());
            }

            err.span_note(multi_span, msg);
        }
        showed = true;
    }
    showed
}

#[derive(Debug)]
struct UsePlacementFinder {
    target_module: NodeId,
    first_legal_span: Option<Span>,
    first_use_span: Option<Span>,
}

impl UsePlacementFinder {
    fn check(krate: &Crate, target_module: NodeId) -> (Option<Span>, FoundUse) {
        let mut finder =
            UsePlacementFinder { target_module, first_legal_span: None, first_use_span: None };
        finder.visit_crate(krate);
        if let Some(use_span) = finder.first_use_span {
            (Some(use_span), FoundUse::Yes)
        } else {
            (finder.first_legal_span, FoundUse::No)
        }
    }
}

impl<'tcx> visit::Visitor<'tcx> for UsePlacementFinder {
    fn visit_crate(&mut self, c: &Crate) {
        if self.target_module == CRATE_NODE_ID {
            let inject = c.spans.inject_use_span;
            if is_span_suitable_for_use_injection(inject) {
                self.first_legal_span = Some(inject);
            }
            self.first_use_span = search_for_any_use_in_items(&c.items);
        } else {
            visit::walk_crate(self, c);
        }
    }

    fn visit_item(&mut self, item: &'tcx ast::Item) {
        if self.target_module == item.id {
            if let ItemKind::Mod(_, _, ModKind::Loaded(items, _inline, mod_spans, _)) = &item.kind {
                let inject = mod_spans.inject_use_span;
                if is_span_suitable_for_use_injection(inject) {
                    self.first_legal_span = Some(inject);
                }
                self.first_use_span = search_for_any_use_in_items(items);
            }
        } else {
            visit::walk_item(self, item);
        }
    }
}

fn search_for_any_use_in_items(items: &[P<ast::Item>]) -> Option<Span> {
    for item in items {
        if let ItemKind::Use(..) = item.kind
            && is_span_suitable_for_use_injection(item.span)
        {
            let mut lo = item.span.lo();
            for attr in &item.attrs {
                if attr.span.eq_ctxt(item.span) {
                    lo = std::cmp::min(lo, attr.span.lo());
                }
            }
            return Some(Span::new(lo, lo, item.span.ctxt(), item.span.parent()));
        }
    }
    None
}

fn is_span_suitable_for_use_injection(s: Span) -> bool {
    // don't suggest placing a use before the prelude
    // import or other generated ones
    !s.from_expansion()
}
