// ignore-tidy-filelength

use std::borrow::Cow;
use std::ops::{ControlFlow, Deref};
use std::{iter, mem};

use itertools::Itertools as _;
use rustc_ast::visit::{self, FnCtxt, FnKind, LifetimeCtxt, Visitor, walk_ty};
use rustc_ast::{
    self as ast, AngleBracketedArg, AssocItemKind, CRATE_NODE_ID, Crate, DUMMY_NODE_ID, Expr,
    ExprKind, GenericArg, GenericArgs, GenericParam, GenericParamKind, Item, ItemKind, MethodCall,
    ModKind, NodeId, Path, PathSegment, QSelf, Ty, TyKind, join_path_idents,
};
use rustc_ast_pretty::pprust::{path_to_string, where_bound_predicate_to_string};
use rustc_attr_parsing::AttributeParser;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::{UnordItems, UnordMap, UnordSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, Diagnostic, ErrorGuaranteed, MultiSpan, SuggestionStyle,
    pluralize, struct_span_code_err,
};
use rustc_feature::BUILTIN_ATTRIBUTES;
use rustc_hir as hir;
use rustc_hir::attrs::diagnostic::{CustomDiagnostic, Directive, FormatArgs};
use rustc_hir::attrs::{AttributeKind, CfgEntry, StrippedCfgItem};
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{CtorKind, CtorOf, DefKind, LifetimeRes, MacroKinds, NonMacroAttrKind, PerNS};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_hir::{Attribute, MissingLifetimeKind, PrimTy, Stability, StabilityLevel, find_attr};
use rustc_middle::bug;
use rustc_middle::ty::{self, TyCtxt, Visibility};
use rustc_session::lint::builtin::{
    ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE, AMBIGUOUS_GLOB_IMPORTS, AMBIGUOUS_IMPORT_VISIBILITIES,
    AMBIGUOUS_PANIC_IMPORTS, MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
};
use rustc_session::utils::was_invoked_from_cargo;
use rustc_session::{Session, lint};
use rustc_span::edit_distance::{edit_distance, find_best_match_for_name};
use rustc_span::edition::Edition;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::SourceMap;
use rustc_span::{
    BytePos, DUMMY_SP, DesugaringKind, Ident, RemapPathScopeComponents, Span, Spanned, Symbol,
    SyntaxContext, kw, sym,
};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, instrument};

use crate::diagnostics::{
    self, AddedMacroUse, ChangeImportBinding, ChangeImportBindingSuggestion, ConsiderAddingADerive,
    ExplicitUnsafeTraits, MacroDefinedLater, MacroRulesNot, MacroSuggMovePosition,
    MaybeMissingMacroRulesName,
};
use crate::hygiene::Macros20NormalizedSyntaxContext;
use crate::imports::{Import, ImportKind, UnresolvedImportError, import_path_to_string};
use crate::late::{
    AliasPossibility, DiagMetadata, LateResolutionVisitor, LifetimeBinderKind, LifetimeRibKind,
    LifetimeUseSet, NoConstantGenericsReason, PatternSource, Rib, RibKind,
};
use crate::ty::fast_reject::SimplifiedType;
use crate::{
    AmbiguityError, AmbiguityKind, AmbiguityWarning, BindingError, BindingKey, Decl, DeclKind,
    DelayedVisResolutionError, Finalize, ForwardGenericParamBanReason, HasGenericParams, IdentKey,
    LateDecl, MacroRulesScope, Module, ModuleKind, ModuleOrUniformRoot, ParentScope, PathResult,
    PathSource, PrivacyError, Res, ResolutionError, Resolver, Scope, ScopeSet, Segment, UseError,
    Used, VisResolutionError, path_names_to_string,
};

/// A vector of spans and replacements, a message and applicability.
pub(crate) type Suggestion = (Vec<(Span, String)>, String, Applicability);

/// Potential candidate for an undeclared or out-of-scope label - contains the ident of a
/// similarly named label and whether or not it is reachable.
pub(crate) type LabelSuggestion = (Ident, bool);

#[derive(Clone)]
pub(crate) struct StructCtor {
    pub res: Res,
    pub vis: Visibility<DefId>,
    pub field_visibilities: Vec<Visibility<DefId>>,
}

impl StructCtor {
    pub(crate) fn has_private_fields<'ra>(&self, m: Module<'ra>, r: &Resolver<'ra, '_>) -> bool {
        self.field_visibilities.iter().any(|&vis| !r.is_accessible_from(vis, m))
    }
}

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
    pub(crate) fn new(candidate: Symbol, span: Span, res: Res) -> TypoSuggestion {
        Self { candidate, span: Some(span), res, target: SuggestionTarget::SimilarlyNamed }
    }
    pub(crate) fn typo_from_name(candidate: Symbol, res: Res) -> TypoSuggestion {
        Self { candidate, span: None, res, target: SuggestionTarget::SimilarlyNamed }
    }
    pub(crate) fn single_item(candidate: Symbol, span: Span, res: Res) -> TypoSuggestion {
        Self { candidate, span: Some(span), res, target: SuggestionTarget::SingleItem }
    }
}

/// A free importable items suggested in case of resolution failure.
#[derive(Debug)]
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
    /// Reports unresolved imports.
    ///
    /// Multiple unresolved import errors within the same use tree are combined into a single
    /// diagnostic.
    pub(crate) fn throw_unresolved_import_error(
        &mut self,
        mut errors: Vec<(Import<'_>, UnresolvedImportError)>,
        glob_error: bool,
    ) {
        errors.retain(|(_import, err)| match err.module {
            // Skip `use` errors for `use foo::Bar;` if `foo.rs` has unrecovered parse errors.
            Some(def_id) if self.mods_with_parse_errors.contains(&def_id) => false,
            // If we've encountered something like `use _;`, we've already emitted an error stating
            // that `_` is not a valid identifier, so we ignore that resolve error.
            _ => err.segment.map(|s| s.name) != Some(kw::Underscore),
        });
        if errors.is_empty() {
            self.tcx.dcx().delayed_bug("expected a parse or \"`_` can't be an identifier\" error");
            return;
        }

        let span = MultiSpan::from_spans(errors.iter().map(|(_, err)| err.span).collect());

        let paths = errors
            .iter()
            .map(|(import, err)| {
                let path = import_path_to_string(
                    &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                    &import.kind,
                    err.span,
                );
                format!("`{path}`")
            })
            .collect::<Vec<_>>();
        let default_message =
            format!("unresolved import{} {}", pluralize!(paths.len()), paths.join(", "),);

        // Process `import` use of  the `#[diagnostic::on_unknown]` attribute.
        //
        // We don't need to check feature gates here; that happens on initialization of the
        // `on_unknown_attr` fields.
        let (mut message, label, mut notes) =
            if let Some(directive) = errors[0].1.on_unknown_attr.as_ref().map(|a| &a.directive) {
                let this = errors
                    .iter()
                    .map(|(_import, err)| {
                        // Is this unwrap_or reachable?
                        err.segment.map(|s| s.name).unwrap_or(kw::Underscore)
                    })
                    .join(", ");

                let args = FormatArgs { unresolved: this.clone(), this, .. };

                let CustomDiagnostic { message, label, notes, parent_label: _dead } =
                    directive.eval(None, &args);

                (message, label, notes)
            } else {
                (None, None, Vec::new())
            };

        // `module` use of the `#[diagnostic::on_unknown]` attribute.
        // We assume that someone who put the attribute on the import has more information than
        // the person who put it on the module, so we choose to prioritize the import attribute.
        let mut mod_diagnostics: Vec<CustomDiagnostic> = errors
            .iter()
            .map(|(import, import_error)| {
                if let Some(ModuleOrUniformRoot::Module(module_data)) = import.imported_module.get()
                    && let ModuleKind::Def(DefKind::Mod, def_id, _, name) = module_data.kind
                {
                    let Some(directive) = self.on_unknown_data(def_id) else {
                        return CustomDiagnostic::default();
                    };

                    let this = if let Some(name) = name {
                        name.to_string()
                    } else if let Some(crate_name) = &self.tcx.sess.opts.crate_name {
                        crate_name.to_string()
                    } else {
                        "<unnamed crate>".to_string()
                    };
                    let unresolved = import_error.segment.map(|s| s.name).unwrap_or(kw::Underscore);
                    let args = FormatArgs { this, unresolved: unresolved.to_string(), .. };

                    directive.eval(None, &args)
                } else {
                    CustomDiagnostic::default()
                }
            })
            .collect();

        // If there is no import attribute with a message,
        // but all mod messages are the same, use that.
        let mod_message =
            mod_diagnostics.iter_mut().flat_map(|d| d.message.take()).all_equal_value();
        if message.is_none()
            && let Ok(mod_msg) = mod_message
        {
            message = Some(mod_msg);
        }

        let mut diag = if let Some(message) = message {
            struct_span_code_err!(self.dcx(), span, E0432, "{message}").with_note(default_message)
        } else {
            struct_span_code_err!(self.dcx(), span, E0432, "{default_message}")
        };

        for mod_diag in mod_diagnostics.iter_mut() {
            for mod_note in mod_diag.notes.drain(..) {
                if !notes.contains(&mod_note) {
                    notes.push(mod_note);
                }
            }
        }

        if !notes.is_empty() {
            for note in notes {
                diag.note(note);
            }
        } else if let Some((_, UnresolvedImportError { note: Some(note), .. })) =
            errors.iter().last()
        {
            diag.note(note.clone());
        }

        /// Upper limit on the number of `span_label` messages.
        const MAX_LABEL_COUNT: usize = 10;
        let mod_labels = mod_diagnostics.into_iter().map(|cd| cd.label);

        for ((import, err), mod_label) in errors.into_iter().zip(mod_labels).take(MAX_LABEL_COUNT) {
            let label_span = match err.segment {
                Some(segment) => segment.span,
                None => err.span,
            };
            if let Some(label) = &label {
                diag.span_label(label_span, label.clone());
            } else if let Some(label) = mod_label {
                diag.span_label(label_span, label);
            } else if let Some(label) = &err.label {
                diag.span_label(label_span, label.clone());
            }

            if let Some((suggestions, msg, applicability)) = err.suggestion {
                if suggestions.is_empty() {
                    diag.help(msg);
                    continue;
                }
                diag.multipart_suggestion(msg, suggestions, applicability);
            }

            if let Some(candidates) = &err.candidates {
                match &import.kind {
                    ImportKind::Single { nested: false, source, target, .. } => import_candidates(
                        self.tcx,
                        &mut diag,
                        Some(err.span),
                        candidates,
                        DiagMode::Import { append: false, unresolved_import: true },
                        (source != target)
                            .then(|| format!(" as {target}"))
                            .as_deref()
                            .unwrap_or(""),
                    ),
                    ImportKind::Single { nested: true, source, target, .. } => {
                        import_candidates(
                            self.tcx,
                            &mut diag,
                            None,
                            candidates,
                            DiagMode::Normal,
                            (source != target)
                                .then(|| format!(" as {target}"))
                                .as_deref()
                                .unwrap_or(""),
                        );
                    }
                    _ => {}
                }
            }

            if matches!(import.kind, ImportKind::Single { .. })
                && let Some(segment) = err.segment
                && let Some(module) = err.module
            {
                self.find_cfg_stripped(&mut diag, &segment.name, module)
            }
        }

        let guar = diag.emit();
        if glob_error {
            self.glob_error = Some(guar);
        }
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }

    pub(crate) fn report_errors(&mut self, krate: &Crate) {
        self.report_delayed_vis_resolution_errors();
        self.report_with_use_injections(krate);

        for &(span_use, span_def) in &self.macro_expanded_macro_export_errors {
            self.lint_buffer.buffer_lint(
                MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
                CRATE_NODE_ID,
                span_use,
                diagnostics::MacroExpandedMacroExportsAccessedByAbsolutePaths {
                    definition: span_def,
                },
            );
        }

        for ambiguity_error in &self.ambiguity_errors {
            let mut diag = self.ambiguity_diagnostic(ambiguity_error);

            if let Some(ambiguity_warning) = ambiguity_error.warning {
                let node_id = match ambiguity_error.b1.0.kind {
                    DeclKind::Import { import, .. } => import.root_id,
                    DeclKind::Def(_) => CRATE_NODE_ID,
                };

                let lint = match ambiguity_warning {
                    _ if ambiguity_error.ambig_vis.is_some() => AMBIGUOUS_IMPORT_VISIBILITIES,
                    AmbiguityWarning::GlobImport => AMBIGUOUS_GLOB_IMPORTS,
                    AmbiguityWarning::PanicImport => AMBIGUOUS_PANIC_IMPORTS,
                };

                self.lint_buffer.buffer_lint(lint, node_id, diag.ident.span, diag);
            } else {
                diag.is_error = true;
                self.dcx().emit_err(diag);
            }
        }

        let mut reported_spans = FxHashSet::default();
        for error in mem::take(&mut self.privacy_errors) {
            if reported_spans.insert(error.dedup_span) {
                self.report_privacy_error(&error);
            }
        }
    }

    fn report_delayed_vis_resolution_errors(&mut self) {
        for DelayedVisResolutionError { vis, parent_scope, error } in
            mem::take(&mut self.delayed_vis_resolution_errors)
        {
            match self.try_resolve_visibility(&parent_scope, &vis, true) {
                Ok(_) => self.report_vis_error(error),
                Err(error) => self.report_vis_error(error),
            };
        }
    }

    fn report_with_use_injections(&mut self, krate: &Crate) {
        for UseError { mut err, candidates, node_id, instead, suggestion, path, is_call } in
            mem::take(&mut self.use_injections)
        {
            let (span, found_use) = if node_id != DUMMY_NODE_ID {
                UsePlacementFinder::check(krate, node_id)
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
        ident: IdentKey,
        ns: Namespace,
        old_binding: Decl<'ra>,
        new_binding: Decl<'ra>,
    ) {
        // Error on the second of two conflicting names
        if old_binding.span.lo() > new_binding.span.lo() {
            return self.report_conflict(ident, ns, new_binding, old_binding);
        }

        let container = match old_binding.parent_module.unwrap().expect_local().kind {
            // Avoid using TyCtxt::def_kind_descr in the resolver, because it
            // indirectly *calls* the resolver, and would cause a query cycle.
            ModuleKind::Def(kind, def_id, _, _) => kind.descr(def_id),
            ModuleKind::Block => "block",
        };

        let (name, span) =
            (ident.name, self.tcx.sess.source_map().guess_head_span(new_binding.span));

        if self.name_already_seen.get(&name) == Some(&span) {
            return;
        }

        let old_kind = match (ns, old_binding.res()) {
            (ValueNS, _) => "value",
            (MacroNS, _) => "macro",
            (TypeNS, _) if old_binding.is_extern_crate() => "extern crate",
            (TypeNS, Res::Def(DefKind::Mod, _)) => "module",
            (TypeNS, Res::Def(DefKind::Trait, _)) => "trait",
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
            true => diagnostics::NameDefinedMultipleTimeLabel::Reimported { span, name },
            false => diagnostics::NameDefinedMultipleTimeLabel::Redefined { span, name },
        };

        let old_binding_label =
            (!old_binding.span.is_dummy() && old_binding.span != span).then(|| {
                let span = self.tcx.sess.source_map().guess_head_span(old_binding.span);
                match old_binding.is_import_user_facing() {
                    true => diagnostics::NameDefinedMultipleTimeOldBindingLabel::Import {
                        span,
                        old_kind,
                        name,
                    },
                    false => diagnostics::NameDefinedMultipleTimeOldBindingLabel::Definition {
                        span,
                        old_kind,
                        name,
                    },
                }
            });

        let mut err = self
            .dcx()
            .create_err(diagnostics::NameDefinedMultipleTime {
                span,
                name,
                descr: ns.descr(),
                container,
                label,
                old_binding_label,
            })
            .with_code(code);

        // See https://github.com/rust-lang/rust/issues/32354
        use DeclKind::Import;
        let can_suggest = |binding: Decl<'_>, import: self::Import<'_>| {
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
            self.extern_prelude.get(&ident).is_none_or(|entry| entry.introduced_by_item());
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
                err.subdiagnostic(diagnostics::ToolOnlyRemoveUnnecessaryImport {
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
                err.subdiagnostic(diagnostics::ToolOnlyRemoveUnnecessaryImport { span });
            } else {
                // Remove the entire line if we cannot extend the span back, this indicates an
                // `issue_52891::{self}` case.
                err.subdiagnostic(diagnostics::RemoveUnnecessaryImport {
                    span: import.use_span_with_attributes,
                });
            }

            return;
        }

        err.subdiagnostic(diagnostics::RemoveUnnecessaryImport { span });
    }

    pub(crate) fn lint_if_path_starts_with_module(
        &mut self,
        finalize: Finalize,
        path: &[Segment],
        second_binding: Option<Decl<'_>>,
    ) {
        let Finalize { node_id, root_span, .. } = finalize;

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
            && let DeclKind::Import { import, .. } = binding.kind
            // Careful: we still want to rewrite paths from renamed extern crates.
            && let ImportKind::ExternCrate { source: None, .. } = import.kind
        {
            return;
        }

        self.lint_buffer.dyn_buffer_lint_any(
            ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
            node_id,
            root_span,
            move |dcx, level, sess| {
                let (replacement, applicability) = match sess
                    .downcast_ref::<Session>()
                    .expect("expected a `Session`")
                    .source_map()
                    .span_to_snippet(root_span)
                {
                    Ok(ref s) => {
                        // FIXME(Manishearth) ideally the emitting code
                        // can tell us whether or not this is global
                        let opt_colon = if s.trim_start().starts_with("::") { "" } else { "::" };

                        (format!("crate{opt_colon}{s}"), Applicability::MachineApplicable)
                    }
                    Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders),
                };
                diagnostics::AbsPathWithModule {
                    sugg: diagnostics::AbsPathWithModuleSugg {
                        span: root_span,
                        applicability,
                        replacement,
                    },
                }
                .into_diag(dcx, level)
            },
        );
    }

    pub(crate) fn add_module_candidates(
        &self,
        module: Module<'ra>,
        names: &mut Vec<TypoSuggestion>,
        filter_fn: &impl Fn(Res) -> bool,
        ctxt: Option<SyntaxContext>,
    ) {
        module.for_each_child(self, |_this, ident, orig_ident_span, _ns, binding| {
            let res = binding.res();
            if filter_fn(res) && ctxt.is_none_or(|ctxt| ctxt == *ident.ctxt) {
                names.push(TypoSuggestion::new(ident.name, orig_ident_span, res));
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
            ResolutionError::GenericParamsFromOuterItem {
                outer_res,
                has_generic_params,
                def_kind,
                inner_item,
                current_self_ty,
            } => {
                use diagnostics::GenericParamsFromOuterItemLabel as Label;
                let static_or_const = match def_kind {
                    DefKind::Static { .. } => {
                        Some(diagnostics::GenericParamsFromOuterItemStaticOrConst::Static)
                    }
                    DefKind::Const { .. } => {
                        Some(diagnostics::GenericParamsFromOuterItemStaticOrConst::Const)
                    }
                    _ => None,
                };
                let is_self =
                    matches!(outer_res, Res::SelfTyParam { .. } | Res::SelfTyAlias { .. });
                let mut err = diagnostics::GenericParamsFromOuterItem {
                    span,
                    label: None,
                    refer_to_type_directly: None,
                    use_let: None,
                    sugg: None,
                    static_or_const,
                    is_self,
                    item: inner_item.as_ref().map(|(label_span, _, kind)| {
                        diagnostics::GenericParamsFromOuterItemInnerItem {
                            span: *label_span,
                            descr: kind.descr().to_string(),
                            is_self,
                        }
                    }),
                };

                let sm = self.tcx.sess.source_map();
                // Note: do not early return for missing def_id here,
                // we still want to provide suggestions for `Res::SelfTyParam` and `Res::SelfTyAlias`.
                let def_id = match outer_res {
                    Res::SelfTyParam { .. } => {
                        err.label = Some(Label::SelfTyParam(span));
                        None
                    }
                    Res::SelfTyAlias { alias_to: def_id, .. } => {
                        err.label = Some(Label::SelfTyAlias(reduce_impl_span_to_impl_keyword(
                            sm,
                            self.def_span(def_id),
                        )));
                        err.refer_to_type_directly = current_self_ty
                            .map(|snippet| diagnostics::UseTypeDirectly { span, snippet });
                        None
                    }
                    Res::Def(DefKind::TyParam, def_id) => {
                        err.label = Some(Label::TyParam(self.def_span(def_id)));
                        Some(def_id)
                    }
                    Res::Def(DefKind::ConstParam, def_id) => {
                        err.label = Some(Label::ConstParam(self.def_span(def_id)));
                        Some(def_id)
                    }
                    _ => {
                        bug!(
                            "GenericParamsFromOuterItem should only be used with \
                            Res::SelfTyParam, Res::SelfTyAlias, DefKind::TyParam or \
                            DefKind::ConstParam"
                        );
                    }
                };

                if let Some((_, item_span, ItemKind::Const(_))) = inner_item.as_ref() {
                    err.use_let = Some(diagnostics::GenericParamsFromOuterItemUseLet {
                        span: sm.span_until_whitespace(*item_span),
                    });
                }

                if let Some(def_id) = def_id
                    && let HasGenericParams::Yes(span) = has_generic_params
                    && !matches!(inner_item, Some((_, _, ItemKind::Delegation(..))))
                {
                    let name = self.tcx.item_name(def_id);
                    let (span, snippet) = if span.is_empty() {
                        let snippet = format!("<{name}>");
                        (span, snippet)
                    } else {
                        let span = sm.span_through_char(span, '<').shrink_to_hi();
                        let snippet = format!("{name}, ");
                        (span, snippet)
                    };
                    err.sugg = Some(diagnostics::GenericParamsFromOuterItemSugg { span, snippet });
                }

                self.dcx().create_err(err)
            }
            ResolutionError::NameAlreadyUsedInParameterList(name, first_use_span) => {
                self.dcx().create_err(diagnostics::NameAlreadyUsedInParameterList {
                    span,
                    first_use_span,
                    name,
                })
            }
            ResolutionError::MethodNotMemberOfTrait(method, trait_, candidate) => {
                self.dcx().create_err(diagnostics::MethodNotMemberOfTrait {
                    span,
                    method,
                    trait_,
                    sub: candidate.map(|c| diagnostics::AssociatedFnWithSimilarNameExists {
                        span: method.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::TypeNotMemberOfTrait(type_, trait_, candidate) => {
                self.dcx().create_err(diagnostics::TypeNotMemberOfTrait {
                    span,
                    type_,
                    trait_,
                    sub: candidate.map(|c| diagnostics::AssociatedTypeWithSimilarNameExists {
                        span: type_.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::ConstNotMemberOfTrait(const_, trait_, candidate) => {
                self.dcx().create_err(diagnostics::ConstNotMemberOfTrait {
                    span,
                    const_,
                    trait_,
                    sub: candidate.map(|c| diagnostics::AssociatedConstWithSimilarNameExists {
                        span: const_.span,
                        candidate: c,
                    }),
                })
            }
            ResolutionError::VariableNotBoundInPattern(binding_error, parent_scope) => {
                let BindingError { name, target, origin, could_be_path } = binding_error;

                let mut target_sp = target.iter().map(|pat| pat.span).collect::<Vec<_>>();
                target_sp.sort();
                target_sp.dedup();
                let mut origin_sp = origin.iter().map(|(span, _)| *span).collect::<Vec<_>>();
                origin_sp.sort();
                origin_sp.dedup();

                let msp = MultiSpan::from_spans(target_sp.clone());
                let mut err = self.dcx().create_err(diagnostics::VariableIsNotBoundInAllPatterns {
                    multispan: msp,
                    name,
                });
                for sp in target_sp {
                    err.subdiagnostic(diagnostics::PatternDoesntBindName { span: sp, name });
                }
                for sp in &origin_sp {
                    err.subdiagnostic(diagnostics::VariableNotInAllPatterns { span: *sp });
                }
                let mut suggested_typo = false;
                if !target.iter().all(|pat| matches!(pat.kind, ast::PatKind::Ident(..)))
                    && !origin.iter().all(|(_, pat)| matches!(pat.kind, ast::PatKind::Ident(..)))
                {
                    // The check above is so that when we encounter `match foo { (a | b) => {} }`,
                    // we don't suggest `(a | a) => {}`, which would never be what the user wants.
                    let mut target_visitor = BindingVisitor::default();
                    for pat in &target {
                        target_visitor.visit_pat(pat);
                    }
                    target_visitor.identifiers.sort();
                    target_visitor.identifiers.dedup();
                    let mut origin_visitor = BindingVisitor::default();
                    for (_, pat) in &origin {
                        origin_visitor.visit_pat(pat);
                    }
                    origin_visitor.identifiers.sort();
                    origin_visitor.identifiers.dedup();
                    // Find if the binding could have been a typo
                    if let Some(typo) =
                        find_best_match_for_name(&target_visitor.identifiers, name.name, None)
                        && !origin_visitor.identifiers.contains(&typo)
                    {
                        err.subdiagnostic(diagnostics::PatternBindingTypo {
                            spans: origin_sp,
                            typo,
                        });
                        suggested_typo = true;
                    }
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
                                        | DefKind::Const { .. }
                                        | DefKind::AssocConst { .. },
                                    _,
                                )
                            )
                        },
                    );

                    if import_suggestions.is_empty() && !suggested_typo {
                        let kind_matches: [fn(DefKind) -> bool; 4] = [
                            |kind| matches!(kind, DefKind::Ctor(CtorOf::Variant, CtorKind::Const)),
                            |kind| matches!(kind, DefKind::Ctor(CtorOf::Struct, CtorKind::Const)),
                            |kind| matches!(kind, DefKind::Const { .. }),
                            |kind| matches!(kind, DefKind::AssocConst { .. }),
                        ];
                        let mut local_names = vec![];
                        self.add_module_candidates(
                            parent_scope.module,
                            &mut local_names,
                            &|res| matches!(res, Res::Def(_, _)),
                            None,
                        );
                        let local_names: FxHashSet<_> = local_names
                            .into_iter()
                            .filter_map(|s| match s.res {
                                Res::Def(_, def_id) => Some(def_id),
                                _ => None,
                            })
                            .collect();

                        let mut local_suggestions = vec![];
                        let mut suggestions = vec![];
                        for matches_kind in kind_matches {
                            if let Some(suggestion) = self.early_lookup_typo_candidate(
                                ScopeSet::All(Namespace::ValueNS),
                                &parent_scope,
                                name,
                                &|res: Res| match res {
                                    Res::Def(k, _) => matches_kind(k),
                                    _ => false,
                                },
                            ) && let Res::Def(kind, mut def_id) = suggestion.res
                            {
                                if let DefKind::Ctor(_, _) = kind {
                                    def_id = self.tcx.parent(def_id);
                                }
                                let kind = kind.descr(def_id);
                                if local_names.contains(&def_id) {
                                    // The item is available in the current scope. Very likely to
                                    // be a typo. Don't use the full path.
                                    local_suggestions.push((
                                        suggestion.candidate,
                                        suggestion.candidate.to_string(),
                                        kind,
                                    ));
                                } else {
                                    suggestions.push((
                                        suggestion.candidate,
                                        self.def_path_str(def_id),
                                        kind,
                                    ));
                                }
                            }
                        }
                        let suggestions = if !local_suggestions.is_empty() {
                            // There is at least one item available in the current scope that is a
                            // likely typo. We only show those.
                            local_suggestions
                        } else {
                            suggestions
                        };
                        for (name, sugg, kind) in suggestions {
                            err.span_suggestion_verbose(
                                span,
                                format!(
                                    "you might have meant to use the similarly named {kind} `{name}`",
                                ),
                                sugg,
                                Applicability::MaybeIncorrect,
                            );
                            suggested_typo = true;
                        }
                    }
                    if import_suggestions.is_empty() && !suggested_typo {
                        let help_msg = format!(
                            "if you meant to match on a unit struct, unit variant or a `const` \
                             item, consider making the path in the pattern qualified: \
                             `path::to::ModOrType::{name}`",
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
                self.dcx().create_err(diagnostics::VariableBoundWithDifferentMode {
                    span,
                    first_binding_span,
                    variable_name,
                })
            }
            ResolutionError::IdentifierBoundMoreThanOnceInParameterList(identifier) => {
                self.dcx().create_err(diagnostics::IdentifierBoundMoreThanOnceInParameterList {
                    span,
                    identifier,
                })
            }
            ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(identifier) => {
                self.dcx().create_err(diagnostics::IdentifierBoundMoreThanOnceInSamePattern {
                    span,
                    identifier,
                })
            }
            ResolutionError::UndeclaredLabel { name, suggestion } => {
                let ((sub_reachable, sub_reachable_suggestion), sub_unreachable) = match suggestion
                {
                    // A reachable label with a similar name exists.
                    Some((ident, true)) => (
                        (
                            Some(diagnostics::LabelWithSimilarNameReachable(ident.span)),
                            Some(diagnostics::TryUsingSimilarlyNamedLabel {
                                span,
                                ident_name: ident.name,
                            }),
                        ),
                        None,
                    ),
                    // An unreachable label with a similar name exists.
                    Some((ident, false)) => (
                        (None, None),
                        Some(diagnostics::UnreachableLabelWithSimilarNameExists {
                            ident_span: ident.span,
                        }),
                    ),
                    // No similarly-named labels exist.
                    None => ((None, None), None),
                };
                self.dcx().create_err(diagnostics::UndeclaredLabel {
                    span,
                    name,
                    sub_reachable,
                    sub_reachable_suggestion,
                    sub_unreachable,
                })
            }
            ResolutionError::FailedToResolve { segment, label, suggestion, module, message } => {
                let mut err = struct_span_code_err!(self.dcx(), span, E0433, "{message}");
                err.span_label(span, label);

                if let Some((suggestions, msg, applicability)) = suggestion {
                    if suggestions.is_empty() {
                        err.help(msg);
                        return err;
                    }
                    err.multipart_suggestion(msg, suggestions, applicability);
                }

                let module = match module {
                    Some(ModuleOrUniformRoot::Module(m)) if let Some(id) = m.opt_def_id() => id,
                    _ => CRATE_DEF_ID.to_def_id(),
                };
                self.find_cfg_stripped(&mut err, &segment, module);

                err
            }
            ResolutionError::CannotCaptureDynamicEnvironmentInFnItem => {
                self.dcx().create_err(diagnostics::CannotCaptureDynamicEnvironmentInFnItem { span })
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

                let (with, with_label, without) = match sp {
                    Some(sp) if !self.tcx.sess.source_map().is_multiline(sp) => {
                        let sp = sp
                            .with_lo(BytePos(sp.lo().0 - (current.len() as u32)))
                            .until(ident.span);

                        // Only suggest replacing the binding keyword if this is a simple
                        // binding.
                        //
                        // Note: this approach still incorrectly suggests for irrefutable
                        // patterns like `if let x = 1 { const { x } }`, since the text
                        // between `let` and the identifier is just whitespace.
                        // See tests/ui/consts/non-const-value-in-const-irrefutable-pat-binding.rs
                        let is_simple_binding =
                            self.tcx.sess.source_map().span_to_snippet(sp).is_ok_and(|snippet| {
                                let after_keyword = snippet[current.len()..].trim();
                                after_keyword.is_empty() || after_keyword == "mut"
                            });

                        if is_simple_binding {
                            (
                                Some(diagnostics::AttemptToUseNonConstantValueInConstantWithSuggestion {
                                    span: sp,
                                    suggestion,
                                    current,
                                    type_span,
                                }),
                                Some(diagnostics::AttemptToUseNonConstantValueInConstantLabelWithSuggestion { span }),
                                None,
                            )
                        } else {
                            (
                                None,
                                Some(diagnostics::AttemptToUseNonConstantValueInConstantLabelWithSuggestion { span }),
                                None,
                            )
                        }
                    }
                    _ => (
                        None,
                        None,
                        Some(
                            diagnostics::AttemptToUseNonConstantValueInConstantWithoutSuggestion {
                                ident_span: ident.span,
                                suggestion,
                            },
                        ),
                    ),
                };

                self.dcx().create_err(diagnostics::AttemptToUseNonConstantValueInConstant {
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
            } => self.dcx().create_err(diagnostics::BindingShadowsSomethingUnacceptable {
                span,
                shadowing_binding,
                shadowed_binding,
                article,
                sub_suggestion: match (shadowing_binding, shadowed_binding) {
                    (
                        PatternSource::Match,
                        Res::Def(DefKind::Ctor(CtorOf::Variant | CtorOf::Struct, CtorKind::Fn), _),
                    ) => Some(diagnostics::BindingShadowsSomethingUnacceptableSuggestion {
                        span,
                        name,
                    }),
                    _ => None,
                },
                shadowed_binding_span,
                participle,
                name,
            }),
            ResolutionError::ForwardDeclaredGenericParam(param, reason) => match reason {
                ForwardGenericParamBanReason::Default => {
                    self.dcx().create_err(diagnostics::ForwardDeclaredGenericParam { param, span })
                }
                ForwardGenericParamBanReason::ConstParamTy => self
                    .dcx()
                    .create_err(diagnostics::ForwardDeclaredGenericInConstParamTy { param, span }),
            },
            ResolutionError::ParamInTyOfConstParam { name } => {
                self.dcx().create_err(diagnostics::ParamInTyOfConstParam { span, name })
            }
            ResolutionError::ParamInNonTrivialAnonConst { is_gca, name, param_kind: is_type } => {
                self.dcx().create_err(diagnostics::ParamInNonTrivialAnonConst {
                    span,
                    name,
                    param_kind: is_type,
                    help: self.tcx.sess.is_nightly_build(),
                    is_gca,
                    help_gca: is_gca,
                })
            }
            ResolutionError::ParamInEnumDiscriminant { name, param_kind: is_type } => {
                self.dcx().create_err(diagnostics::ParamInEnumDiscriminant {
                    span,
                    name,
                    param_kind: is_type,
                })
            }
            ResolutionError::ForwardDeclaredSelf(reason) => match reason {
                ForwardGenericParamBanReason::Default => {
                    self.dcx().create_err(diagnostics::SelfInGenericParamDefault { span })
                }
                ForwardGenericParamBanReason::ConstParamTy => {
                    self.dcx().create_err(diagnostics::SelfInConstGenericTy { span })
                }
            },
            ResolutionError::UnreachableLabel { name, definition_span, suggestion } => {
                let ((sub_suggestion_label, sub_suggestion), sub_unreachable_label) =
                    match suggestion {
                        // A reachable label with a similar name exists.
                        Some((ident, true)) => (
                            (
                                Some(diagnostics::UnreachableLabelSubLabel {
                                    ident_span: ident.span,
                                }),
                                Some(diagnostics::UnreachableLabelSubSuggestion {
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
                            Some(diagnostics::UnreachableLabelSubLabelUnreachable {
                                ident_span: ident.span,
                            }),
                        ),
                        // No similarly-named labels exist.
                        None => ((None, None), None),
                    };
                self.dcx().create_err(diagnostics::UnreachableLabel {
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
                .create_err(diagnostics::TraitImplMismatch {
                    span,
                    name,
                    kind,
                    trait_path,
                    trait_item_span,
                })
                .with_code(code),
            ResolutionError::TraitImplDuplicate { name, trait_item_span, old_span } => {
                self.dcx().create_err(diagnostics::TraitImplDuplicate {
                    span,
                    name,
                    trait_item_span,
                    old_span,
                })
            }
            ResolutionError::InvalidAsmSym => {
                self.dcx().create_err(diagnostics::InvalidAsmSym { span })
            }
            ResolutionError::LowercaseSelf => {
                self.dcx().create_err(diagnostics::LowercaseSelf { span })
            }
            ResolutionError::BindingInNeverPattern => {
                self.dcx().create_err(diagnostics::BindingInNeverPattern { span })
            }
        }
    }

    pub(crate) fn report_vis_error(
        &mut self,
        vis_resolution_error: VisResolutionError,
    ) -> ErrorGuaranteed {
        match vis_resolution_error {
            VisResolutionError::Relative2018(span, path) => {
                self.dcx().create_err(diagnostics::Relative2018 {
                    span,
                    path_span: path.span,
                    // intentionally converting to String, as the text would also be used as
                    // in suggestion context
                    path_str: path_to_string(&path),
                })
            }
            VisResolutionError::AncestorOnly(span) => {
                self.dcx().create_err(diagnostics::AncestorOnly(span))
            }
            VisResolutionError::FailedToResolve(span, segment, label, suggestion, message) => self
                .into_struct_error(
                    span,
                    ResolutionError::FailedToResolve {
                        segment,
                        label,
                        suggestion,
                        module: None,
                        message,
                    },
                ),
            VisResolutionError::ExpectedFound(span, path_str, res) => {
                self.dcx().create_err(diagnostics::ExpectedModuleFound { span, res, path_str })
            }
            VisResolutionError::Indeterminate(span) => {
                self.dcx().create_err(diagnostics::Indeterminate(span))
            }
            VisResolutionError::ModuleOnly(span) => {
                self.dcx().create_err(diagnostics::ModuleOnly(span))
            }
        }
        .emit()
    }

    pub(crate) fn def_path_str(&self, mut def_id: DefId) -> String {
        // We can't use `def_path_str` in resolve.
        let mut path = vec![def_id];
        while let Some(parent) = self.tcx.opt_parent(def_id) {
            def_id = parent;
            path.push(def_id);
            if def_id.is_top_level_module() {
                break;
            }
        }
        // We will only suggest importing directly if it is accessible through that path.
        path.into_iter()
            .rev()
            .map(|def_id| {
                self.tcx
                    .opt_item_name(def_id)
                    .map(|name| {
                        match (
                            def_id.is_top_level_module(),
                            def_id.is_local(),
                            self.tcx.sess.edition(),
                        ) {
                            (true, true, Edition::Edition2015) => String::new(),
                            (true, true, _) => kw::Crate.to_string(),
                            (true, false, _) | (false, _, _) => name.to_string(),
                        }
                    })
                    .unwrap_or_else(|| "_".to_string())
            })
            .collect::<Vec<String>>()
            .join("::")
    }

    pub(crate) fn add_scope_set_candidates(
        &mut self,
        suggestions: &mut Vec<TypoSuggestion>,
        scope_set: ScopeSet<'ra>,
        ps: &ParentScope<'ra>,
        sp: Span,
        filter_fn: &impl Fn(Res) -> bool,
    ) {
        let ctxt = Macros20NormalizedSyntaxContext::new(sp.ctxt());
        self.cm().visit_scopes(scope_set, ps, ctxt, sp, None, |this, scope, use_prelude, _| {
            match scope {
                Scope::DeriveHelpers(expn_id) => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);
                    if filter_fn(res) {
                        suggestions.extend(this.helper_attrs.get(&expn_id).into_flat_iter().map(
                            |&(ident, orig_ident_span, _)| {
                                TypoSuggestion::new(ident.name, orig_ident_span, res)
                            },
                        ));
                    }
                }
                Scope::DeriveHelpersCompat => {
                    // Never recommend deprecated helper attributes.
                }
                Scope::MacroRules(macro_rules_scope) => {
                    if let MacroRulesScope::Def(macro_rules_def) = macro_rules_scope.get() {
                        let res = macro_rules_def.decl.res();
                        if filter_fn(res) {
                            suggestions.push(TypoSuggestion::new(
                                macro_rules_def.ident.name,
                                macro_rules_def.orig_ident_span,
                                res,
                            ))
                        }
                    }
                }
                Scope::ModuleNonGlobs(module, _) => {
                    this.add_module_candidates(module, suggestions, filter_fn, None);
                }
                Scope::ModuleGlobs(..) => {
                    // Already handled in `ModuleNonGlobs`.
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
                                // These trace attributes are compiler-generated and have
                                // deliberately invalid names.
                                .filter(|attr| {
                                    !matches!(**attr, sym::cfg_trace | sym::cfg_attr_trace)
                                })
                                .map(|attr| TypoSuggestion::typo_from_name(*attr, res)),
                        );
                    }
                }
                Scope::ExternPreludeItems => {
                    // Add idents from both item and flag scopes.
                    suggestions.extend(this.extern_prelude.iter().filter_map(|(ident, entry)| {
                        let res = Res::Def(DefKind::Mod, CRATE_DEF_ID.to_def_id());
                        filter_fn(res).then_some(TypoSuggestion::new(ident.name, entry.span(), res))
                    }));
                }
                Scope::ExternPreludeFlags => {}
                Scope::ToolPrelude => {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                    suggestions.extend(
                        this.registered_tools
                            .iter()
                            .map(|ident| TypoSuggestion::new(ident.name, ident.span, res)),
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

            ControlFlow::<()>::Continue(())
        });
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
        self.add_scope_set_candidates(
            &mut suggestions,
            scope_set,
            parent_scope,
            ident.span,
            filter_fn,
        );

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
        &self,
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
            in_module.for_each_child(self, |this, ident, orig_ident_span, ns, name_binding| {
                // Avoid non-importable candidates.
                if name_binding.is_assoc_item()
                    && !this.features.import_trait_associated_functions()
                {
                    return;
                }

                if ident.name == kw::Underscore {
                    return;
                }

                let child_accessible =
                    accessible && this.is_accessible_from(name_binding.vis(), parent_scope.module);

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
                if let DeclKind::Import { source_decl, .. } = name_binding.kind
                    && this.is_accessible_from(source_decl.vis(), parent_scope.module)
                    && !this.is_accessible_from(name_binding.vis(), parent_scope.module)
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
                    && ident.ctxt.is_root()
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

                    segms.push(ast::PathSegment::from_ident(ident.orig(orig_ident_span)));
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
                                && find_attr!(
                                    this.tcx,
                                    did,
                                    RustcDiagnosticItem(
                                        sym::TryInto | sym::TryFrom | sym::FromIterator
                                    )
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
                if let Some(def_id) = name_binding.res().module_like_def_id() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident.orig(orig_ident_span)));

                    let alias_import = if let DeclKind::Import { import, .. } = name_binding.kind
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
                        if seen_modules.insert(def_id) {
                            if via_import { &mut worklist_via_import } else { &mut worklist }.push(
                                (
                                    this.expect_module(def_id),
                                    path_segments,
                                    child_accessible,
                                    child_doc_visible,
                                    is_stable && this.is_stable(def_id, name_binding.span),
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
                level: StabilityLevel::Unstable { implied_by, .. }, feature, ..
            }) => {
                if span.allows_unstable(feature) {
                    true
                } else if self.features.enabled(feature) {
                    true
                } else if let Some(implied_by) = implied_by
                    && self.features.enabled(implied_by)
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
            self.graph_root.to_module(),
            crate_path,
            &filter_fn,
        );

        if lookup_ident.span.at_least_rust_2018() {
            for (ident, entry) in &self.extern_prelude {
                if entry.span().from_expansion() {
                    // Idents are adjusted to the root context before being
                    // resolved in the extern prelude, so reporting this to the
                    // user is no help. This skips the injected
                    // `extern crate std` in the 2018 edition, which would
                    // otherwise cause duplicate suggestions.
                    continue;
                }
                let Some(crate_id) =
                    self.cstore_mut().maybe_process_path_extern(self.tcx, ident.name)
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
                                && key.ident == *ident
                                && let Some(decl) = name_resolution.borrow().best_decl()
                            {
                                match decl.res() {
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
                crate_path.push(ast::PathSegment::from_ident(ident.orig(entry.span())));

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

        suggestions.retain(|suggestion| suggestion.is_stable || self.tcx.sess.is_nightly_build());
        suggestions
    }

    pub(crate) fn unresolved_macro_suggestions(
        &mut self,
        err: &mut Diag<'_>,
        macro_kind: MacroKind,
        parent_scope: &ParentScope<'ra>,
        ident: Ident,
        krate: &Crate,
        sugg_span: Option<Span>,
    ) {
        // Bring all unused `derive` macros into `macro_map` so we ensure they can be used for
        // suggestions.
        self.register_macros_for_all_crates();

        let is_expected =
            &|res: Res| res.macro_kinds().is_some_and(|k| k.contains(macro_kind.into()));
        let suggestion = self.early_lookup_typo_candidate(
            ScopeSet::Macro(macro_kind),
            parent_scope,
            ident,
            is_expected,
        );
        if !self.add_typo_suggestion(err, suggestion, ident.span) {
            self.detect_derive_attribute(err, ident, parent_scope, sugg_span);
        }

        let import_suggestions =
            self.lookup_import_candidates(ident, Namespace::MacroNS, parent_scope, is_expected);
        let (span, found_use) = match parent_scope.module.nearest_parent_mod_node_id() {
            DUMMY_NODE_ID => (None, FoundUse::No),
            node_id => UsePlacementFinder::check(krate, node_id),
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
            let unused_macro_kinds = self.local_macro_map[def_id].macro_kinds();
            if !unused_macro_kinds.contains(macro_kind.into()) {
                match macro_kind {
                    MacroKind::Bang => {
                        err.subdiagnostic(MacroRulesNot::Func { span: unused_ident.span, ident });
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
            if Some(parent_nearest) == scope.opt_def_id() {
                err.subdiagnostic(MacroDefinedLater { span: unused_ident.span });
                err.subdiagnostic(MacroSuggMovePosition { span: ident.span, ident });
                return;
            }
        }

        if ident.name == kw::Default
            && let ModuleKind::Def(DefKind::Enum, def_id, _, _) = parent_scope.module.kind
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
            let Ok(binding) = self.cm().resolve_ident_in_scope_set(
                ident,
                ScopeSet::All(ns),
                parent_scope,
                None,
                None,
                None,
            ) else {
                continue;
            };

            let desc = match binding.res() {
                Res::Def(DefKind::Macro(MacroKinds::BANG), _) => {
                    "a function-like macro".to_string()
                }
                Res::Def(DefKind::Macro(MacroKinds::ATTR), _) | Res::NonMacroAttr(..) => {
                    format!("an attribute: `#[{ident}]`")
                }
                Res::Def(DefKind::Macro(MacroKinds::DERIVE), _) => {
                    format!("a derive macro: `#[derive({ident})]`")
                }
                Res::Def(DefKind::Macro(kinds), _) => {
                    format!("{} {}", kinds.article(), kinds.descr())
                }
                Res::ToolMod | Res::OpenMod(..) => {
                    // Don't confuse the user with tool modules or open modules.
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
            if let crate::DeclKind::Import { import, .. } = binding.kind
                && !import.span.is_dummy()
            {
                let note = diagnostics::IdentImporterHereButItIsDesc {
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
            let note = diagnostics::IdentInScopeButItIsDesc {
                imported_ident: ident,
                imported_ident_desc: &desc,
            };
            err.subdiagnostic(note);
            return;
        }

        if self.macro_names.contains(&IdentKey::new(ident)) {
            err.subdiagnostic(AddedMacroUse);
            return;
        }
    }

    /// Given an attribute macro that failed to be resolved, look for `derive` macros that could
    /// provide it, either as-is or with small typos.
    fn detect_derive_attribute(
        &self,
        err: &mut Diag<'_>,
        ident: Ident,
        parent_scope: &ParentScope<'ra>,
        sugg_span: Option<Span>,
    ) {
        // Find all of the `derive`s in scope and collect their corresponding declared
        // attributes.
        // FIXME: this only works if the crate that owns the macro that has the helper_attr
        // has already been imported.
        let mut derives = vec![];
        let mut all_attrs: UnordMap<Symbol, Vec<_>> = UnordMap::default();
        // We're collecting these in a hashmap, and handle ordering the output further down.
        #[allow(rustc::potential_query_instability)]
        for (def_id, ext) in self
            .local_macro_map
            .iter()
            .map(|(local_id, ext)| (local_id.to_def_id(), ext))
            .chain(self.extern_macro_map.borrow().iter().map(|(id, d)| (*id, d)))
        {
            for helper_attr in &ext.helper_attrs {
                let item_name = self.tcx.item_name(def_id);
                all_attrs.entry(*helper_attr).or_default().push(item_name);
                if helper_attr == &ident.name {
                    derives.push(item_name);
                }
            }
        }
        let kind = MacroKind::Derive.descr();
        if !derives.is_empty() {
            // We found an exact match for the missing attribute in a `derive` macro. Suggest it.
            let mut derives: Vec<String> = derives.into_iter().map(|d| d.to_string()).collect();
            derives.sort();
            derives.dedup();
            let msg = match &derives[..] {
                [derive] => format!(" `{derive}`"),
                [start @ .., last] => format!(
                    "s {} and `{last}`",
                    start.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(", ")
                ),
                [] => unreachable!("we checked for this to be non-empty 10 lines above!?"),
            };
            let msg = format!(
                "`{}` is an attribute that can be used by the {kind}{msg}, you might be \
                     missing a `derive` attribute",
                ident.name,
            );
            let sugg_span =
                if let ModuleKind::Def(DefKind::Enum, id, _, _) = parent_scope.module.kind {
                    let span = self.def_span(id);
                    if span.from_expansion() {
                        None
                    } else {
                        // For enum variants sugg_span is empty but we can get the enum's Span.
                        Some(span.shrink_to_lo())
                    }
                } else {
                    // For items this `Span` will be populated, everything else it'll be None.
                    sugg_span
                };
            match sugg_span {
                Some(span) => {
                    err.span_suggestion_verbose(
                        span,
                        msg,
                        format!("#[derive({})]\n", derives.join(", ")),
                        Applicability::MaybeIncorrect,
                    );
                }
                None => {
                    err.note(msg);
                }
            }
        } else {
            // We didn't find an exact match. Look for close matches. If any, suggest fixing typo.
            let all_attr_names = all_attrs.keys().map(|s| *s).into_sorted_stable_ord();
            if let Some(best_match) = find_best_match_for_name(&all_attr_names, ident.name, None)
                && let Some(macros) = all_attrs.get(&best_match)
            {
                let mut macros: Vec<String> = macros.into_iter().map(|d| d.to_string()).collect();
                macros.sort();
                macros.dedup();
                let msg = match &macros[..] {
                    [] => return,
                    [name] => format!(" `{name}` accepts"),
                    [start @ .., end] => format!(
                        "s {} and `{end}` accept",
                        start.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", "),
                    ),
                };
                let msg = format!("the {kind}{msg} the similarly named `{best_match}` attribute");
                err.span_suggestion_verbose(
                    ident.span,
                    msg,
                    best_match,
                    Applicability::MaybeIncorrect,
                );
            }
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
                //   --> $DIR/unicode-string-literal-syntax-error-64792.rs:4:14
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
                    diagnostics::DefinedHere::SimilarlyNamed { span, candidate_descr, candidate }
                }
                SuggestionTarget::SingleItem => {
                    diagnostics::DefinedHere::SingleItem { span, candidate_descr, candidate }
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
        err.span_suggestion_verbose(span, msg, sugg, Applicability::MaybeIncorrect);
        true
    }

    fn decl_description(&self, b: Decl<'_>, ident: Ident, scope: Scope<'_>) -> String {
        let res = b.res();
        if b.span.is_dummy() || !self.tcx.sess.source_map().is_span_accessible(b.span) {
            let (built_in, from) = match scope {
                Scope::StdLibPrelude | Scope::MacroUsePrelude => ("", " from prelude"),
                Scope::ExternPreludeFlags
                    if self.tcx.sess.opts.externs.get(ident.as_str()).is_some()
                        || matches!(res, Res::OpenMod(..)) =>
                {
                    ("", " passed with `--extern`")
                }
                _ => {
                    if matches!(res, Res::NonMacroAttr(..) | Res::PrimTy(..) | Res::ToolMod) {
                        // These already contain the "built-in" prefix or look bad with it.
                        ("", "")
                    } else {
                        (" built-in", "")
                    }
                }
            };

            let a = if built_in.is_empty() { res.article() } else { "a" };
            format!("{a}{built_in} {thing}{from}", thing = res.descr())
        } else {
            let introduced = if b.is_import_user_facing() { "imported" } else { "defined" };
            format!("the {thing} {introduced} here", thing = res.descr())
        }
    }

    fn ambiguity_diagnostic(
        &self,
        ambiguity_error: &AmbiguityError<'ra>,
    ) -> diagnostics::Ambiguity {
        let AmbiguityError { kind, ambig_vis, ident, b1, b2, scope1, scope2, .. } =
            *ambiguity_error;
        let extern_prelude_ambiguity = || {
            // Note: b1 may come from a module scope, as an extern crate item in module.
            matches!(scope2, Scope::ExternPreludeFlags)
                && self
                    .extern_prelude
                    .get(&IdentKey::new(ident))
                    .is_some_and(|entry| entry.item_decl.map(|(b, ..)| b) == Some(b1))
        };
        let (b1, b2, scope1, scope2, swapped) = if b2.span.is_dummy() && !b1.span.is_dummy() {
            // We have to print the span-less alternative first, otherwise formatting looks bad.
            (b2, b1, scope2, scope1, true)
        } else {
            (b1, b2, scope1, scope2, false)
        };

        let could_refer_to = |b: Decl<'_>, scope: Scope<'ra>, also: &str| {
            let what = self.decl_description(b, ident, scope);
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
            if b.is_extern_crate() && ident.span.at_least_rust_2018() && !extern_prelude_ambiguity()
            {
                help_msgs.push(format!("use `::{ident}` to refer to this {thing} unambiguously"))
            }

            if kind != AmbiguityKind::GlobVsGlob {
                if let Scope::ModuleNonGlobs(module, _) | Scope::ModuleGlobs(module, _) = scope {
                    if module == self.graph_root.to_module() {
                        help_msgs.push(format!(
                            "use `crate::{ident}` to refer to this {thing} unambiguously"
                        ));
                    } else if module.is_normal() {
                        help_msgs.push(format!(
                            "use `self::{ident}` to refer to this {thing} unambiguously"
                        ));
                    }
                }
            }

            (
                Spanned { node: note_msg, span: b.span },
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
        let (b1_note, b1_help_msgs) = could_refer_to(b1, scope1, "");
        let (b2_note, b2_help_msgs) = could_refer_to(b2, scope2, " also");
        let help = if kind == AmbiguityKind::GlobVsGlob
            && b1
                .parent_module
                .and_then(|m| m.opt_def_id())
                .map(|d| !d.is_local())
                .unwrap_or_default()
        {
            Some(&[
                "consider updating this dependency to resolve this error",
                "if updating the dependency does not resolve the problem report the problem to the author of the relevant crate",
            ] as &[_])
        } else {
            None
        };

        let ambig_vis = ambig_vis.map(|(vis1, vis2)| {
            format!(
                "{} or {}",
                vis1.to_string(CRATE_DEF_ID, self.tcx),
                vis2.to_string(CRATE_DEF_ID, self.tcx)
            )
        });

        diagnostics::Ambiguity {
            ident,
            help,
            ambig_vis,
            kind: kind.descr(),
            b1_note,
            b1_help_msgs,
            b2_note,
            b2_help_msgs,
            is_error: false,
        }
    }

    /// If the binding refers to a tuple struct constructor with fields,
    /// returns the span of its fields.
    fn ctor_fields_span(&self, decl: Decl<'_>) -> Option<Span> {
        let DeclKind::Def(Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), ctor_def_id)) =
            decl.kind
        else {
            return None;
        };

        let def_id = self.tcx.parent(ctor_def_id);
        self.field_idents(def_id)?.iter().map(|&f| f.span).reduce(Span::to) // None for `struct Foo()`
    }

    /// Returns the path segments (as symbols) of a module, including `kw::Crate` at the start.
    /// For example, for `crate::foo::bar`, returns `[Crate, foo, bar]`.
    /// Returns `None` for block modules that don't have a `DefId`.
    fn module_path_names(&self, module: Module<'ra>) -> Option<Vec<Symbol>> {
        let mut path = Vec::new();
        let mut def_id = module.opt_def_id()?;
        while let Some(parent) = self.tcx.opt_parent(def_id) {
            if let Some(name) = self.tcx.opt_item_name(def_id) {
                path.push(name);
            }
            if parent.is_top_level_module() {
                break;
            }
            def_id = parent;
        }
        path.reverse();
        path.insert(0, kw::Crate);
        Some(path)
    }

    /// Shortens a candidate import path to use `super::` (up to 1 level) or `self::` (same module)
    /// relative to the current scope, if possible. Only applies to crate-local items and
    /// only when the resulting path is actually shorter than the original.
    fn shorten_candidate_path(
        &self,
        suggestion: &mut ImportSuggestion,
        current_module: Module<'ra>,
    ) {
        const MAX_SUPER_PATH_ITEMS_IN_SUGGESTION: usize = 1;

        // Only shorten local items.
        if suggestion.did.is_none_or(|did| !did.is_local()) {
            return;
        }

        // Build current module path: [Crate, foo, bar, ...].
        let Some(current_mod_path) = self.module_path_names(current_module) else {
            return;
        };

        // Normalise candidate path: filter out `PathRoot` (`::`), and if the path
        // doesn't start with `Crate`, prepend it (edition 2015 paths are relative
        // to the crate root without an explicit `crate::` prefix).
        let candidate_names = {
            let filtered_segments: Vec<_> = suggestion
                .path
                .segments
                .iter()
                .filter(|segment| segment.ident.name != kw::PathRoot)
                .collect();

            let mut candidate_names: Vec<Symbol> =
                filtered_segments.iter().map(|segment| segment.ident.name).collect();
            if candidate_names.first() != Some(&kw::Crate) {
                candidate_names.insert(0, kw::Crate);
            }
            if candidate_names.len() < 2 {
                return;
            }
            candidate_names
        };

        // The candidate's module path is everything except the last segment (the item name).
        let candidate_mod_names = &candidate_names[..candidate_names.len() - 1];

        // Find the longest common prefix between the current module and candidate module paths.
        let common_prefix_length = current_mod_path
            .iter()
            .zip(candidate_mod_names.iter())
            .take_while(|(current, candidate)| current == candidate)
            .count();

        // Non-crate-local item; keep the full absolute path.
        if common_prefix_length == 0 {
            return;
        }

        let super_count = current_mod_path.len() - common_prefix_length;

        // At the crate root, `use` paths resolve from the crate root anyway, so we can
        // drop the `crate::` prefix entirely instead of replacing it with `self::`.
        let at_crate_root = current_mod_path.len() == 1;

        let mut new_segments = if super_count == 0 && at_crate_root {
            ThinVec::new()
        } else {
            let prefix_keyword = match super_count {
                0 => kw::SelfLower,
                1..=MAX_SUPER_PATH_ITEMS_IN_SUGGESTION => kw::Super,
                _ => return, // Too many `super` levels; keep the full absolute path.
            };
            thin_vec![ast::PathSegment::from_ident(Ident::with_dummy_span(prefix_keyword),)]
        };
        for &name in &candidate_names[common_prefix_length..] {
            new_segments.push(ast::PathSegment::from_ident(Ident::with_dummy_span(name)));
        }

        // Only apply if the result is strictly shorter than the original path.
        if new_segments.len() >= suggestion.path.segments.len() {
            return;
        }

        suggestion.path = Path { span: suggestion.path.span, segments: new_segments, tokens: None };
    }

    fn report_privacy_error(&mut self, privacy_error: &PrivacyError<'ra>) {
        let PrivacyError {
            ident,
            decl,
            outermost_res,
            parent_scope,
            single_nested,
            dedup_span,
            ref source,
        } = *privacy_error;

        let res = decl.res();
        let ctor_fields_span = self.ctor_fields_span(decl);
        let plain_descr = res.descr().to_string();
        let nonimport_descr =
            if ctor_fields_span.is_some() { plain_descr + " constructor" } else { plain_descr };
        let import_descr = nonimport_descr.clone() + " import";
        let get_descr = |b: Decl<'_>| if b.is_import() { &import_descr } else { &nonimport_descr };

        // Print the primary message.
        let ident_descr = get_descr(decl);
        let mut err =
            self.dcx().create_err(diagnostics::IsPrivate { span: ident.span, ident_descr, ident });

        self.mention_default_field_values(source, ident, &mut err);

        let shown_candidates = if let Some((this_res, outer_ident)) = outermost_res {
            let mut import_suggestions = self.lookup_import_candidates(
                outer_ident,
                this_res.ns().unwrap_or(Namespace::TypeNS),
                &parent_scope,
                &|res: Res| res == this_res,
            );
            // Shorten candidate paths using `super::` or `self::` when possible.
            for suggestion in &mut import_suggestions {
                self.shorten_candidate_path(suggestion, parent_scope.module);
            }
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
                let label = diagnostics::OuterIdentIsNotPubliclyReexported {
                    span: outer_ident.span,
                    outer_ident_descr: this_res.descr(),
                    outer_ident,
                };
                err.subdiagnostic(label);
            }
            !point_to_def
        } else {
            false
        };

        let mut non_exhaustive = None;
        // If an ADT is foreign and marked as `non_exhaustive`, then that's
        // probably why we have the privacy error.
        // Otherwise, point out if the struct has any private fields.
        if let Some(def_id) = res.opt_def_id()
            && !def_id.is_local()
            && let Some(attr_span) = find_attr!(self.tcx, def_id, NonExhaustive(span) => *span)
        {
            non_exhaustive = Some(attr_span);
        } else if let Some(span) = ctor_fields_span {
            let label = diagnostics::ConstructorPrivateIfAnyFieldPrivate { span };
            err.subdiagnostic(label);
            if let Res::Def(_, d) = res
                && let Some(fields) = self.field_visibility_spans.get(&d)
            {
                let spans = fields.iter().map(|span| *span).collect();
                let sugg = diagnostics::ConsiderMakingTheFieldPublic {
                    spans,
                    number_of_fields: fields.len(),
                };
                err.subdiagnostic(sugg);
            }
        }

        let mut sugg_paths: Vec<(Vec<Ident>, bool)> = vec![];
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
            let path_names: Option<Vec<Ident>> = path
                .iter()
                .rev()
                .map(|def_id| {
                    self.tcx.opt_item_name(*def_id).map(|name| {
                        Ident::with_dummy_span(if def_id.is_top_level_module() {
                            kw::Crate
                        } else {
                            name
                        })
                    })
                })
                .collect();
            if let Some(&def_id) = path.get(0)
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
        let first_binding = decl;
        let mut next_binding = Some(decl);
        let mut next_ident = ident;
        while let Some(binding) = next_binding {
            let name = next_ident;
            next_binding = match binding.kind {
                _ if res == Res::Err => None,
                DeclKind::Import { source_decl, import, .. } => match import.kind {
                    _ if source_decl.span.is_dummy() => None,
                    ImportKind::Single { source, .. } => {
                        next_ident = source;
                        Some(source_decl)
                    }
                    ImportKind::Glob { .. }
                    | ImportKind::MacroUse { .. }
                    | ImportKind::MacroExport => Some(source_decl),
                    ImportKind::ExternCrate { .. } => None,
                },
                _ => None,
            };

            match binding.kind {
                DeclKind::Import { source_decl, import, .. } => {
                    // Don't include `{{root}}` in suggestions - it's an internal symbol
                    // that should never be shown to users.
                    let path = import
                        .module_path
                        .iter()
                        .filter(|seg| seg.ident.name != kw::PathRoot)
                        .map(|seg| seg.ident.clone())
                        .chain(std::iter::once(ident))
                        .collect::<Vec<_>>();
                    let through_reexport = !matches!(source_decl.kind, DeclKind::Def(_));
                    sugg_paths.push((path, through_reexport));
                }
                DeclKind::Def(_) => {}
            }
            let first = binding == first_binding;
            let def_span = self.tcx.sess.source_map().guess_head_span(binding.span);
            let mut note_span = MultiSpan::from_span(def_span);
            if !first && binding.vis().is_public() {
                let desc = match binding.kind {
                    DeclKind::Import { .. } => "re-export",
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
            let note = diagnostics::NoteAndRefersToTheItemDefinedHere {
                span: note_span,
                binding_descr: get_descr(binding),
                binding_name: name,
                first,
                dots: next_binding.is_some(),
            };
            err.subdiagnostic(note);
        }
        // The suggestion replaces `dedup_span` with a path reaching the failing ident.
        // That's valid only when
        // 1) the failing ident is the imported leaf, otherwise `as` renames and trailing segments
        //    get dropped, and
        // 2) the use isn't nested, otherwise `dedup_span` is one ident in `{...}`.
        //
        // See issue #156060.
        let can_replace_use = !shown_candidates
            && !single_nested
            && !outermost_res.is_some_and(|(_, outer)| outer.span != ident.span);
        if can_replace_use {
            // We prioritize shorter paths, non-core imports and direct imports over the
            // alternatives.
            sugg_paths.sort_by_key(|(p, reexport)| (p.len(), p[0].name == sym::core, *reexport));
            for (sugg, reexport) in sugg_paths {
                if sugg.len() <= 1 {
                    // A single path segment suggestion is wrong. This happens on circular
                    // imports. `tests/ui/imports/issue-55884-2.rs`
                    continue;
                }
                let path = join_path_idents(sugg);
                let sugg = if reexport {
                    diagnostics::ImportIdent::ThroughReExport { span: dedup_span, ident, path }
                } else {
                    diagnostics::ImportIdent::Directly { span: dedup_span, ident, path }
                };
                err.subdiagnostic(sugg);
                break;
            }
        }

        err.emit();
    }

    /// When a private field is being set that has a default field value, we suggest using `..` and
    /// setting the value of that field implicitly with its default.
    ///
    /// If we encounter code like
    /// ```text
    /// struct Priv;
    /// pub struct S {
    ///     pub field: Priv = Priv,
    /// }
    /// ```
    /// which is used from a place where `Priv` isn't accessible
    /// ```text
    /// let _ = S { field: m::Priv1 {} };
    /// //                    ^^^^^ private struct
    /// ```
    /// we will suggest instead using the `default_field_values` syntax instead:
    /// ```text
    /// let _ = S { .. };
    /// ```
    fn mention_default_field_values(
        &self,
        source: &Option<ast::Expr>,
        ident: Ident,
        err: &mut Diag<'_>,
    ) {
        let Some(expr) = source else { return };
        let ast::ExprKind::Struct(struct_expr) = &expr.kind else { return };
        // We don't have to handle type-relative paths because they're forbidden in ADT
        // expressions, but that would change with `#[feature(more_qualified_paths)]`.
        let Some(segment) = struct_expr.path.segments.last() else { return };
        let Some(partial_res) = self.partial_res_map.get(&segment.id) else { return };
        let Some(Res::Def(_, def_id)) = partial_res.full_res() else {
            return;
        };
        let Some(default_fields) = self.field_defaults(def_id) else { return };
        if struct_expr.fields.is_empty() {
            return;
        }
        let last_span = struct_expr.fields.iter().last().unwrap().span;
        let mut iter = struct_expr.fields.iter().peekable();
        let mut prev: Option<Span> = None;
        while let Some(field) = iter.next() {
            if field.expr.span.overlaps(ident.span) {
                err.span_label(field.ident.span, "while setting this field");
                if default_fields.contains(&field.ident.name) {
                    let sugg = if last_span == field.span {
                        vec![(field.span, "..".to_string())]
                    } else {
                        vec![
                            (
                                // Account for trailing commas and ensure we remove them.
                                match (prev, iter.peek()) {
                                    (_, Some(next)) => field.span.with_hi(next.span.lo()),
                                    (Some(prev), _) => field.span.with_lo(prev.hi()),
                                    (None, None) => field.span,
                                },
                                String::new(),
                            ),
                            (last_span.shrink_to_hi(), ", ..".to_string()),
                        ]
                    };
                    err.multipart_suggestion(
                        format!(
                            "the type `{ident}` of field `{}` is private, but you can construct \
                             the default value defined for it in `{}` using `..` in the struct \
                             initializer expression",
                            field.ident,
                            self.tcx.item_name(def_id),
                        ),
                        sugg,
                        Applicability::MachineApplicable,
                    );
                    break;
                }
            }
            prev = Some(field.span);
        }
    }

    pub(crate) fn find_similarly_named_module_or_crate(
        &self,
        ident: Symbol,
        current_module: Module<'ra>,
    ) -> Option<Symbol> {
        let mut candidates = self
            .extern_prelude
            .keys()
            .map(|ident| ident.name)
            .chain(
                self.local_module_map
                    .iter()
                    .filter(|(_, module)| {
                        let module = module.to_module();
                        current_module.is_ancestor_of(module) && current_module != module
                    })
                    .flat_map(|(_, module)| module.name()),
            )
            .chain(
                self.extern_module_map
                    .borrow()
                    .iter()
                    .filter(|(_, module)| {
                        let module = module.to_module();
                        current_module.is_ancestor_of(module) && current_module != module
                    })
                    .flat_map(|(_, module)| module.name()),
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
        ignore_decl: Option<Decl<'ra>>,
        ignore_import: Option<Import<'ra>>,
        module: Option<ModuleOrUniformRoot<'ra>>,
        failed_segment_idx: usize,
        ident: Ident,
        diag_metadata: Option<&DiagMetadata<'_>>,
    ) -> (String, String, Option<Suggestion>) {
        let is_last = failed_segment_idx == path.len() - 1;
        let ns = if is_last { opt_ns.unwrap_or(TypeNS) } else { TypeNS };
        let module_def_id = match module {
            Some(ModuleOrUniformRoot::Module(module)) => module.opt_def_id(),
            _ => None,
        };
        let scope = match &path[..failed_segment_idx] {
            [.., prev] => {
                if prev.ident.name == kw::PathRoot {
                    format!("the crate root")
                } else {
                    format!("`{}`", prev.ident)
                }
            }
            _ => format!("this scope"),
        };
        let message = format!("cannot find `{ident}` in {scope}");

        if module_def_id == Some(CRATE_DEF_ID.to_def_id()) {
            let is_mod = |res| matches!(res, Res::Def(DefKind::Mod, _));
            let mut candidates = self.lookup_import_candidates(ident, TypeNS, parent_scope, is_mod);
            candidates.sort_by_cached_key(|c| (c.path.segments.len(), path_to_string(&c.path)));
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
                    message,
                    String::from("unresolved import"),
                    Some((
                        vec![(ident.span, path_to_string(&path))],
                        String::from("a similar path exists"),
                        Applicability::MaybeIncorrect,
                    )),
                )
            } else if ident.name == sym::core {
                (
                    message,
                    format!("you might be missing crate `{ident}`"),
                    Some((
                        vec![(ident.span, "std".to_string())],
                        "try using `std` instead of `core`".to_string(),
                        Applicability::MaybeIncorrect,
                    )),
                )
            } else if ident.name == kw::Underscore {
                (
                    "invalid crate or module name `_`".to_string(),
                    "`_` is not a valid crate or module name".to_string(),
                    None,
                )
            } else if self.tcx.sess.is_rust_2015() {
                (
                    format!("cannot find module or crate `{ident}` in {scope}"),
                    format!("use of unresolved module or unlinked crate `{ident}`"),
                    Some((
                        vec![(
                            self.current_crate_outer_attr_insert_span,
                            format!("extern crate {ident};\n"),
                        )],
                        if was_invoked_from_cargo() {
                            format!(
                                "if you wanted to use a crate named `{ident}`, use `cargo add \
                                 {ident}` to add it to your `Cargo.toml` and import it in your \
                                 code",
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
                (message, format!("could not find `{ident}` in the crate root"), None)
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
                    self.cm()
                        .resolve_ident_in_module(
                            module,
                            ident,
                            ns_to_try,
                            parent_scope,
                            None,
                            ignore_decl,
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
                        ignore_decl,
                        diag_metadata,
                    ) {
                        // we found a locally-imported or available item/module
                        Some(LateDecl::Decl(binding)) => Some(binding),
                        _ => None,
                    }
                } else {
                    self.cm()
                        .resolve_ident_in_scope_set(
                            ident,
                            ScopeSet::All(ns_to_try),
                            parent_scope,
                            None,
                            ignore_decl,
                            ignore_import,
                        )
                        .ok()
                };
                if let Some(binding) = binding {
                    msg = format!(
                        "expected {}, found {} `{ident}` in {parent}",
                        ns.descr(),
                        binding.res().descr(),
                    );
                };
            }
            (message, msg, None)
        } else if ident.name == kw::SelfUpper {
            // As mentioned above, `opt_ns` being `None` indicates a module path in import.
            // We can use this to improve a confusing error for, e.g. `use Self::Variant` in an
            // impl
            if opt_ns.is_none() {
                (message, "`Self` cannot be used in imports".to_string(), None)
            } else {
                (
                    message,
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
                    ignore_decl,
                    diag_metadata,
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
                Some(LateDecl::RibDef(Res::Local(id))) => {
                    Some((*self.pat_span_map.get(&id).unwrap(), "a", "local binding"))
                }
                // Name matches item from a local name binding
                // created by `use` declaration. For example:
                // ```
                // pub const Foo: &str = "";
                //
                // mod submod {
                //     use super::Foo;
                //     println!("{}", Foo::Bar); // Name refers to local
                //                               // binding `Foo`.
                // }
                // ```
                Some(LateDecl::Decl(name_binding)) => Some((
                    name_binding.span,
                    name_binding.res().article(),
                    name_binding.res().descr(),
                )),
                _ => None,
            };

            let message = format!("cannot find type `{ident}` in {scope}");
            let label = if let Some((span, article, descr)) = match_span {
                format!(
                    "`{ident}` is declared as {article} {descr} at `{}`, not a type",
                    self.tcx
                        .sess
                        .source_map()
                        .span_to_short_string(span, RemapPathScopeComponents::DIAGNOSTICS)
                )
            } else {
                format!("use of undeclared type `{ident}`")
            };
            (message, label, None)
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
            if let Ok(binding) = self.cm().resolve_ident_in_scope_set(
                ident,
                ScopeSet::All(ValueNS),
                parent_scope,
                None,
                ignore_decl,
                ignore_import,
            ) {
                let descr = binding.res().descr();
                let message = format!("cannot find module or crate `{ident}` in {scope}");
                (message, format!("{descr} `{ident}` is not a crate or module"), suggestion)
            } else {
                let suggestion = if suggestion.is_some() {
                    suggestion
                } else if let Some(m) = self.undeclared_module_exists(ident) {
                    self.undeclared_module_suggest_declare(ident, m)
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
                let message = format!("cannot find module or crate `{ident}` in {scope}");
                (
                    message,
                    format!("use of unresolved module or unlinked crate `{ident}`"),
                    suggestion,
                )
            }
        }
    }

    fn undeclared_module_suggest_declare(
        &self,
        ident: Ident,
        path: std::path::PathBuf,
    ) -> Option<(Vec<(Span, String)>, String, Applicability)> {
        Some((
            vec![(self.current_crate_outer_attr_insert_span, format!("mod {ident};\n"))],
            format!(
                "to make use of source file {}, use `mod {ident}` \
                 in this file to declare the module",
                path.display()
            ),
            Applicability::MaybeIncorrect,
        ))
    }

    fn undeclared_module_exists(&self, ident: Ident) -> Option<std::path::PathBuf> {
        let map = self.tcx.sess.source_map();

        let src = map.span_to_filename(ident.span).into_local_path()?;
        let i = ident.as_str();
        // FIXME: add case where non parent using undeclared module (hard?)
        let dir = src.parent()?;
        let src = src.file_stem()?.to_str()?;
        for file in [
            // …/x.rs
            dir.join(i).with_extension("rs"),
            // …/x/mod.rs
            dir.join(i).join("mod.rs"),
        ] {
            if file.exists() {
                return Some(file);
            }
        }
        if !matches!(src, "main" | "lib" | "mod") {
            for file in [
                // …/x/y.rs
                dir.join(src).join(i).with_extension("rs"),
                // …/x/y/mod.rs
                dir.join(src).join(i).join("mod.rs"),
            ] {
                if file.exists() {
                    return Some(file);
                }
            }
        }
        None
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
        let result = self.cm().maybe_resolve_path(&path, None, parent_scope, None);
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
        let result = self.cm().maybe_resolve_path(&path, None, parent_scope, None);
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
        let result = self.cm().maybe_resolve_path(&path, None, parent_scope, None);
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
            let result = self.cm().maybe_resolve_path(&path, None, parent_scope, None);
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

        let binding_key = BindingKey::new(IdentKey::new(ident), MacroNS);
        let binding = self.resolution(crate_module, binding_key)?.best_decl()?;
        let Res::Def(DefKind::Macro(kinds), _) = binding.res() else {
            return None;
        };
        if !kinds.contains(MacroKinds::BANG) {
            return None;
        }
        let module_name = crate_module.name().unwrap_or(kw::Crate);
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
            //   i.e. `use a::b::{c, d, e};`
            //                      ^^^
            let (found_closing_brace, binding_span) = find_span_of_binding_until_next_binding(
                self.tcx.sess,
                import.span,
                import.use_span,
            );
            debug!(found_closing_brace, ?binding_span);

            let mut removal_span = binding_span;

            // If the binding span ended with a closing brace, as in the below example:
            //   i.e. `use a::b::{c, d};`
            //                      ^
            // Then expand the span of characters to remove to include the previous
            // binding's trailing comma.
            //   i.e. `use a::b::{c, d};`
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
            //   i.e. `use a::b::{c, d};`
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
                    let parent_scope = self.local_modules.iter().find_map(|m| match m.kind {
                        ModuleKind::Def(_, def_id, node_id, _) if node_id == item.parent_scope => {
                            Some(def_id)
                        }
                        _ => None,
                    })?;
                    Some(StrippedCfgItem { parent_scope, ident: item.ident, cfg: item.cfg.clone() })
                })
                .collect::<Vec<_>>();
            local_items.as_slice()
        } else {
            self.tcx.stripped_cfg_items(module.krate)
        };

        for &StrippedCfgItem { parent_scope, ident, ref cfg } in symbols {
            if ident.name != *segment {
                continue;
            }

            let parent_module = self.get_nearest_non_block_module(parent_scope).def_id();

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
                let mut res = false;
                let m = r.expect_module(parent_module);
                if m.is_local() {
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
                                res = true;
                                break;
                            }
                        }
                    }
                }
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

            let item_was = if let CfgEntry::NameValue { value: Some(feature), .. } = cfg.0 {
                diagnostics::ItemWas::BehindFeature { feature, span: cfg.1 }
            } else {
                diagnostics::ItemWas::CfgOut { span: cfg.1 }
            };
            let note = diagnostics::FoundItemConfigureOut { span: ident.span, item_was };
            err.subdiagnostic(note);
        }
    }

    pub(crate) fn struct_ctor(&self, def_id: DefId) -> Option<StructCtor> {
        match def_id.as_local() {
            Some(def_id) => self.struct_ctors.get(&def_id).cloned(),
            None => {
                self.cstore().ctor_untracked(self.tcx, def_id).map(|(ctor_kind, ctor_def_id)| {
                    let res = Res::Def(DefKind::Ctor(CtorOf::Struct, ctor_kind), ctor_def_id);
                    let vis = self.tcx.visibility(ctor_def_id);
                    let field_visibilities = self
                        .tcx
                        .associated_item_def_ids(def_id)
                        .iter()
                        .map(|&field_id| self.tcx.visibility(field_id))
                        .collect();
                    StructCtor { res, vis, field_visibilities }
                })
            }
        }
    }

    /// Gets the `#[diagnostic::on_unknown]` attribute data associated with this `DefId`.
    fn on_unknown_data(&self, def_id: DefId) -> Option<&Directive> {
        match def_id.as_local() {
            Some(local) => Some(self.on_unknown_data.get(&local)?.directive.as_ref()),
            None => find_attr!(self.tcx, def_id, OnUnknown{ directive } => directive)?.as_deref(),
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
    //   i.e. `a, e};` or `a};`
    let binding_until_end = binding_span.with_hi(use_span.hi());

    // Find everything after the binding but not including the binding.
    //   i.e. `, e};` or `};`
    let after_binding_until_end = binding_until_end.with_lo(binding_span.hi());

    // Keep characters in the span until we encounter something that isn't a comma or
    // whitespace.
    //   i.e. `, ` or ``.
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
    //   i.e. `a, ` or `a`.
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
    // Then split based on a command and take the first (i.e. closest to our span)
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
                    path_to_string(&c.path),
                    c.descr,
                    c.did.and_then(|did| Some(tcx.source_span(did.as_local()?))),
                    &c.note,
                    c.via_import,
                ))
            }
        } else {
            inaccessible_path_strings.push((
                path_to_string(&c.path),
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
            let descr = inaccessible_path_strings
                .iter()
                .map(|&(_, descr, _, _, _)| descr)
                .all_equal_value()
                .unwrap_or("item");
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

impl<'tcx> Visitor<'tcx> for UsePlacementFinder {
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
            if let ItemKind::Mod(_, _, ModKind::Loaded(items, _inline, mod_spans)) = &item.kind {
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

#[derive(Default)]
struct BindingVisitor {
    identifiers: Vec<Symbol>,
    spans: FxHashMap<Symbol, Vec<Span>>,
}

impl<'tcx> Visitor<'tcx> for BindingVisitor {
    fn visit_pat(&mut self, pat: &ast::Pat) {
        if let ast::PatKind::Ident(_, ident, _) = pat.kind {
            self.identifiers.push(ident.name);
            self.spans.entry(ident.name).or_default().push(ident.span);
        }
        visit::walk_pat(self, pat);
    }
}

fn search_for_any_use_in_items(items: &[Box<ast::Item>]) -> Option<Span> {
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

#[derive(Debug, Clone, Default)]
pub(crate) struct OnUnknownData {
    pub(crate) directive: Box<Directive>,
}

impl OnUnknownData {
    pub(crate) fn from_attrs(
        r: &Resolver<'_, '_>,
        attrs: &[ast::Attribute],
    ) -> Option<OnUnknownData> {
        if r.features.diagnostic_on_unknown()
            && let Some(Attribute::Parsed(AttributeKind::OnUnknown { directive, .. })) =
                AttributeParser::parse_limited(
                    r.tcx.sess,
                    attrs,
                    &[sym::diagnostic, sym::on_unknown],
                )
        {
            Some(Self { directive: directive? })
        } else {
            None
        }
    }
}

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

fn path_to_string_without_assoc_item_bindings(path: &Path) -> String {
    let mut path = path.clone();
    for segment in &mut path.segments {
        let mut remove_args = false;
        if let Some(args) = segment.args.as_deref_mut()
            && let ast::GenericArgs::AngleBracketed(angle_bracketed) = args
        {
            angle_bracketed.args.retain(|arg| matches!(arg, ast::AngleBracketedArg::Arg(_)));
            remove_args = angle_bracketed.args.is_empty();
        }
        if remove_args {
            segment.args = None;
        }
    }
    path_to_string(&path)
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
pub(crate) struct MissingLifetime {
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
#[derive(Debug)]
pub(crate) struct ElisionFnParameter {
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
#[derive(Clone, Copy, Debug)]
pub(crate) enum LifetimeElisionCandidate {
    /// This is not a real lifetime, or it is a named lifetime, in which case we won't suggest anything.
    Ignore,
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

impl<'ast, 'ra, 'tcx> LateResolutionVisitor<'_, 'ast, 'ra, 'tcx> {
    fn trait_assoc_type_def_id_by_name(
        &mut self,
        trait_def_id: DefId,
        assoc_name: Symbol,
    ) -> Option<DefId> {
        let module = self.r.get_module(trait_def_id)?;
        self.r.resolutions(module).borrow().iter().find_map(|(key, resolution)| {
            if key.ident.name != assoc_name {
                return None;
            }
            let resolution = resolution.borrow();
            let binding = resolution.best_decl()?;
            match binding.res() {
                Res::Def(DefKind::AssocTy, def_id) => Some(def_id),
                _ => None,
            }
        })
    }

    /// This does best-effort work to generate suggestions for associated types.
    fn suggest_assoc_type_from_bounds(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_, 'ast, 'ra>,
        path: &[Segment],
        ident_span: Span,
    ) -> bool {
        // Filter out cases where we cannot emit meaningful suggestions.
        if source.namespace() != TypeNS {
            return false;
        }
        let [segment] = path else { return false };
        if segment.has_generic_args {
            return false;
        }
        if !ident_span.can_be_used_for_suggestions() {
            return false;
        }
        let assoc_name = segment.ident.name;
        if assoc_name == kw::Underscore {
            return false;
        }

        // Map: type parameter name -> (trait def id -> (assoc type def id, trait paths as written)).
        // We keep a set of paths per trait so we can detect cases like
        // `T: Trait<i32> + Trait<u32>` where suggesting `T::Assoc` would be ambiguous.
        let mut matching_bounds: FxIndexMap<
            Symbol,
            FxIndexMap<DefId, (DefId, FxIndexSet<String>)>,
        > = FxIndexMap::default();

        let mut record_bound = |this: &mut Self,
                                ty_param: Symbol,
                                poly_trait_ref: &ast::PolyTraitRef| {
            // Avoid generating suggestions we can't print in a well-formed way.
            if !poly_trait_ref.bound_generic_params.is_empty() {
                return;
            }
            if poly_trait_ref.modifiers != ast::TraitBoundModifiers::NONE {
                return;
            }
            let Some(trait_seg) = poly_trait_ref.trait_ref.path.segments.last() else {
                return;
            };
            let Some(partial_res) = this.r.partial_res_map.get(&trait_seg.id) else {
                return;
            };
            let Some(trait_def_id) = partial_res.full_res().and_then(|res| res.opt_def_id()) else {
                return;
            };
            let Some(assoc_type_def_id) =
                this.trait_assoc_type_def_id_by_name(trait_def_id, assoc_name)
            else {
                return;
            };

            // Preserve `::` and generic args so we don't generate broken suggestions like
            // `<T as Foo>::Assoc` for bounds written as `T: ::Foo<'a>`, while stripping
            // associated-item bindings that are rejected in qualified paths.
            let trait_path =
                path_to_string_without_assoc_item_bindings(&poly_trait_ref.trait_ref.path);
            let trait_bounds = matching_bounds.entry(ty_param).or_default();
            let trait_bounds = trait_bounds
                .entry(trait_def_id)
                .or_insert_with(|| (assoc_type_def_id, FxIndexSet::default()));
            debug_assert_eq!(trait_bounds.0, assoc_type_def_id);
            trait_bounds.1.insert(trait_path);
        };

        let mut record_from_generics = |this: &mut Self, generics: &ast::Generics| {
            for param in &generics.params {
                let ast::GenericParamKind::Type { .. } = param.kind else { continue };
                for bound in &param.bounds {
                    let ast::GenericBound::Trait(poly_trait_ref) = bound else { continue };
                    record_bound(this, param.ident.name, poly_trait_ref);
                }
            }

            for predicate in &generics.where_clause.predicates {
                let ast::WherePredicateKind::BoundPredicate(where_bound) = &predicate.kind else {
                    continue;
                };

                let ast::TyKind::Path(None, bounded_path) = &where_bound.bounded_ty.kind else {
                    continue;
                };
                let [ast::PathSegment { ident, args: None, .. }] = &bounded_path.segments[..]
                else {
                    continue;
                };

                // Only suggest for bounds that are explicitly on an in-scope type parameter.
                let Some(partial_res) = this.r.partial_res_map.get(&where_bound.bounded_ty.id)
                else {
                    continue;
                };
                if !matches!(partial_res.full_res(), Some(Res::Def(DefKind::TyParam, _))) {
                    continue;
                }

                for bound in &where_bound.bounds {
                    let ast::GenericBound::Trait(poly_trait_ref) = bound else { continue };
                    record_bound(this, ident.name, poly_trait_ref);
                }
            }
        };

        if let Some(item) = self.diag_metadata.current_item
            && let Some(generics) = item.kind.generics()
        {
            record_from_generics(self, generics);
        }

        if let Some(item) = self.diag_metadata.current_item
            && matches!(item.kind, ItemKind::Impl(..))
            && let Some(assoc) = self.diag_metadata.current_impl_item
        {
            let generics = match &assoc.kind {
                AssocItemKind::Const(ast::ConstItem { generics, .. })
                | AssocItemKind::Fn(ast::Fn { generics, .. })
                | AssocItemKind::Type(ast::TyAlias { generics, .. }) => Some(generics),
                AssocItemKind::Delegation(..)
                | AssocItemKind::MacCall(..)
                | AssocItemKind::DelegationMac(..) => None,
            };
            if let Some(generics) = generics {
                record_from_generics(self, generics);
            }
        }

        let mut suggestions: FxIndexSet<String> = FxIndexSet::default();
        for (ty_param, traits) in matching_bounds {
            let ty_param = ty_param.to_ident_string();
            let trait_paths_len: usize = traits.values().map(|(_, paths)| paths.len()).sum();
            if traits.len() == 1 && trait_paths_len == 1 {
                let assoc_type_def_id = traits.values().next().unwrap().0;
                let assoc_segment = format!(
                    "{}{}",
                    assoc_name,
                    self.r.item_required_generic_args_suggestion(assoc_type_def_id)
                );
                suggestions.insert(format!("{ty_param}::{assoc_segment}"));
            } else {
                for (assoc_type_def_id, trait_paths) in traits.into_values() {
                    let assoc_segment = format!(
                        "{}{}",
                        assoc_name,
                        self.r.item_required_generic_args_suggestion(assoc_type_def_id)
                    );
                    for trait_path in trait_paths {
                        suggestions
                            .insert(format!("<{ty_param} as {trait_path}>::{assoc_segment}"));
                    }
                }
            }
        }

        if suggestions.is_empty() {
            return false;
        }

        let mut suggestions: Vec<String> = suggestions.into_iter().collect();
        suggestions.sort();

        err.span_suggestions_with_style(
            ident_span,
            "you might have meant to use an associated type of the same name",
            suggestions,
            Applicability::MaybeIncorrect,
            SuggestionStyle::ShowAlways,
        );

        true
    }

    fn make_base_error(
        &mut self,
        path: &[Segment],
        span: Span,
        source: PathSource<'_, 'ast, 'ra>,
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
                        DefKind::Ctor(..)
                        | DefKind::AssocFn
                        | DefKind::Const { .. }
                        | DefKind::AssocConst { .. },
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
                let mod_res = self.resolve_path(mod_path, Some(TypeNS), None, source);
                let mod_prefix = match mod_res {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => module.res(),
                    _ => None,
                };

                let module_did = mod_prefix.as_ref().and_then(Res::mod_def_id);

                let mod_prefix =
                    mod_prefix.map_or_else(String::new, |res| format!("{} ", res.descr()));
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
    #[tracing::instrument(skip(self), level = "debug")]
    pub(crate) fn smart_resolve_report_errors(
        &mut self,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        source: PathSource<'_, 'ast, 'ra>,
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
        self.suggest_range_struct_destructuring(&mut err, path, source);
        self.suggest_swapping_misplaced_self_ty_and_trait(&mut err, source, res, base_error.span);

        if let Some((span, label)) = base_error.span_label {
            err.span_label(span, label);
        }

        if let Some(ref sugg) = base_error.suggestion {
            err.span_suggestion_verbose(sugg.0, sugg.1, &sugg.2, Applicability::MaybeIncorrect);
        }

        self.suggest_changing_type_to_const_param(&mut err, res, source, path, following_seg, span);
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

        if let Some((did, item)) = self.lookup_doc_alias_name(path, source.namespace()) {
            let item_name = item.name;
            let suggestion_name = self.r.tcx.item_name(did);
            err.span_suggestion(
                item.span,
                format!("`{suggestion_name}` has a name defined in the doc alias attribute as `{item_name}`"),
                    suggestion_name,
                    Applicability::MaybeIncorrect
                );

            return (err, Vec::new());
        };

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

        let module = base_error.module.unwrap_or_else(|| CRATE_DEF_ID.to_def_id());
        self.r.find_cfg_stripped(&mut err, &path.last().unwrap().ident.name, module);

        (err, candidates)
    }

    fn detect_rtn_with_fully_qualified_path(
        &self,
        err: &mut Diag<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        source: PathSource<'_, '_, '_>,
        res: Option<Res>,
        qself: Option<&QSelf>,
    ) {
        if let Some(Res::Def(DefKind::AssocFn, _)) = res
            && let PathSource::TraitItem(TypeNS, _) = source
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
        source: PathSource<'_, '_, '_>,
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

                // Contextualize for E0425 "cannot find type", but don't belabor the point
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
                            let source_map = self.r.tcx.sess.source_map();
                            let field_is_format_named_arg = matches!(
                                span.desugaring_kind(),
                                Some(DesugaringKind::FormatLiteral { .. })
                            ) && source_map
                                .span_to_source(span, |s, start, _| {
                                    Ok(s.get(start.saturating_sub(1)..start) == Some("{"))
                                })
                                .unwrap_or(false);
                            if field_is_format_named_arg {
                                err.help(
                                    format!("you might have meant to use the available field in a format string: `\"{{}}\", self.{}`", segment.ident.name),
                                );
                            } else {
                                err.span_suggestion_verbose(
                                    span.shrink_to_lo(),
                                    "you might have meant to use the available field",
                                    format!("{pre}self."),
                                    Applicability::MaybeIncorrect,
                                );
                            }
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
                if let Some(args_span) = args_span
                    && let Ok(snippet) = self.r.tcx.sess.source_map().span_to_snippet(args_span)
                {
                    args_snippet = snippet;
                }

                if let Some(Res::Def(DefKind::Struct, def_id)) = res {
                    if let Some(ctor) = self.r.struct_ctor(def_id)
                        && ctor.has_private_fields(self.parent_scope.module, self.r)
                    {
                        if matches!(
                            ctor.res,
                            Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), _)
                        ) {
                            self.update_err_for_private_tuple_struct_fields(err, &source, def_id);
                        }
                        err.note("constructor is not visible here due to private fields");
                    }
                } else {
                    err.span_suggestion(
                        call_span,
                        format!("try calling `{ident}` as a method"),
                        format!("self.{path_str}({args_snippet})"),
                        Applicability::MachineApplicable,
                    );
                }

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
        if let Some(rib) = &self.last_block_rib {
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

    fn lookup_doc_alias_name(&mut self, path: &[Segment], ns: Namespace) -> Option<(DefId, Ident)> {
        let find_doc_alias_name = |r: &mut Resolver<'ra, '_>, m: Module<'ra>, item_name: Symbol| {
            for resolution in r.resolutions(m).borrow().values() {
                let Some(did) =
                    resolution.borrow().best_decl().and_then(|binding| binding.res().opt_def_id())
                else {
                    continue;
                };
                if did.is_local() {
                    // We don't record the doc alias name in the local crate
                    // because the people who write doc alias are usually not
                    // confused by them.
                    continue;
                }
                if let Some(d) = hir::find_attr!(r.tcx, did, Doc(d) => d)
                    && d.aliases.contains_key(&item_name)
                {
                    return Some(did);
                }
            }
            None
        };

        if path.len() == 1 {
            for rib in self.ribs[ns].iter().rev() {
                let item = path[0].ident;
                if let RibKind::Module(module) | RibKind::Block(Some(module)) = rib.kind
                    && let Some(did) = find_doc_alias_name(self.r, module.to_module(), item.name)
                {
                    return Some((did, item));
                }
            }
        } else {
            // Finds to the last resolved module item in the path
            // and searches doc aliases within that module.
            //
            // Example: For the path `a::b::last_resolved::not_exist::c::d`,
            // we will try to find any item has doc aliases named `not_exist`
            // in `last_resolved` module.
            //
            // - Use `skip(1)` because the final segment must remain unresolved.
            for (idx, seg) in path.iter().enumerate().rev().skip(1) {
                let Some(id) = seg.id else {
                    continue;
                };
                let Some(res) = self.r.partial_res_map.get(&id) else {
                    continue;
                };
                if let Res::Def(DefKind::Mod, module) = res.expect_full_res()
                    && let module = self.r.expect_module(module)
                    && let item = path[idx + 1].ident
                    && let Some(did) = find_doc_alias_name(self.r, module, item.name)
                {
                    return Some((did, item));
                }
                break;
            }
        }
        None
    }

    fn suggest_trait_and_bounds(
        &self,
        err: &mut Diag<'_>,
        source: PathSource<'_, '_, '_>,
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
            && let Some(bounds @ [first_bound, .., last_bound]) =
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

        fallback |= self.restrict_assoc_type_in_where_clause(span, err);
        fallback
    }

    fn suggest_typo(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_, 'ast, 'ra>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
        base_error: &BaseError,
        suggested_candidates: FxHashSet<String>,
    ) -> bool {
        let is_expected = &|res| source.is_expected(res);
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);

        // Prefer suggestions based on associated types from in-scope bounds (e.g. `T::Item`)
        // over purely edit-distance-based identifier suggestions.
        // Otherwise suggestions could be verbose.
        if self.suggest_assoc_type_from_bounds(err, source, path, ident_span) {
            return false;
        }

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
                    err.span_suggestion_verbose(
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
        source: PathSource<'_, '_, '_>,
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
        source: PathSource<'_, '_, '_>,
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

                self.suggest_ident_hidden_by_hygiene(err, path, span);
                // cannot find type in this scope
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

    fn suggest_ident_hidden_by_hygiene(&self, err: &mut Diag<'_>, path: &[Segment], span: Span) {
        let [segment] = path else { return };

        let ident = segment.ident;
        let callsite_span = span.source_callsite();
        for rib in self.ribs[ValueNS].iter().rev() {
            for (binding_ident, _) in &rib.bindings {
                // Case 1: the identifier is defined in the same scope as the macro is called
                if binding_ident.name == ident.name
                    && !binding_ident.span.eq_ctxt(span)
                    && !binding_ident.span.from_expansion()
                    && binding_ident.span.lo() < callsite_span.lo()
                {
                    err.span_help(
                        binding_ident.span,
                        "an identifier with the same name exists, but is not accessible due to macro hygiene",
                    );
                    return;
                }

                // Case 2: the identifier is defined in a macro call in the same scope
                if binding_ident.name == ident.name
                    && binding_ident.span.from_expansion()
                    && binding_ident.span.source_callsite().eq_ctxt(callsite_span)
                    && binding_ident.span.source_callsite().lo() < callsite_span.lo()
                {
                    err.span_help(
                        binding_ident.span,
                        "an identifier with the same name is defined here, but is not accessible due to macro hygiene",
                    );
                    return;
                }
            }
        }
    }

    /// Emit special messages for unresolved `Self` and `self`.
    fn suggest_self_ty(
        &self,
        err: &mut Diag<'_>,
        source: PathSource<'_, '_, '_>,
        path: &[Segment],
        span: Span,
    ) -> bool {
        if !is_self_type(path, source.namespace()) {
            return false;
        }
        err.code(E0411);
        err.span_label(span, "`Self` is only available in impls, traits, and type definitions");
        if let Some(item) = self.diag_metadata.current_item
            && let Some(ident) = item.kind.ident()
        {
            err.span_label(
                ident.span,
                format!("`Self` not allowed in {} {}", item.kind.article(), item.kind.descr()),
            );
        }
        true
    }

    fn suggest_self_value(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_, '_, '_>,
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

        // using `let self` is wrong even if we're not in an associated method or if we're in a macro expansion.
        // So, we should return early if we're in a pattern, see issue #143134.
        if matches!(source, PathSource::Pat) {
            return true;
        }

        let is_assoc_fn = self.self_type_is_available();
        let self_from_macro = "a `self` parameter, but a macro invocation can only \
                               access identifiers it receives from parameters";
        if let Some((fn_kind, fn_span)) = &self.diag_metadata.current_function {
            // The current function has a `self` parameter, but we were unable to resolve
            // a reference to `self`. This can only happen if the `self` identifier we
            // are resolving came from a different hygiene context or a variable binding.
            // But variable binding error is returned early above.
            if fn_kind.decl().inputs.get(0).is_some_and(|p| p.is_self()) {
                err.span_label(*fn_span, format!("this function has {self_from_macro}"));
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
                                .map_or(*fn_span, |ident| fn_span.with_lo(ident.span.hi()));
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
        &self,
        err: &mut Diag<'_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
    ) {
        let [segment] = path else { return };
        let None = following_seg else { return };
        for rib in self.ribs[ValueNS].iter().rev() {
            let patterns_with_skipped_bindings =
                self.r.tcx.with_stable_hashing_context(|mut hcx| {
                    rib.patterns_with_skipped_bindings.to_sorted(&mut hcx, true)
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

    fn suggest_at_operator_in_slice_pat_with_range(&self, err: &mut Diag<'_>, path: &[Segment]) {
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
            err.subdiagnostic(diagnostics::UnexpectedResUseAtOpInSlicePatWithRangeSugg {
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

    fn suggest_range_struct_destructuring(
        &mut self,
        err: &mut Diag<'_>,
        path: &[Segment],
        source: PathSource<'_, '_, '_>,
    ) {
        if !matches!(source, PathSource::Pat | PathSource::TupleStruct(..) | PathSource::Expr(..)) {
            return;
        }

        let Some(pat) = self.diag_metadata.current_pat else { return };
        let ast::PatKind::Range(start, end, end_kind) = &pat.kind else { return };

        let [segment] = path else { return };
        let failing_span = segment.ident.span;

        let in_start = start.as_ref().is_some_and(|e| e.span.contains(failing_span));
        let in_end = end.as_ref().is_some_and(|e| e.span.contains(failing_span));

        if !in_start && !in_end {
            return;
        }

        let start_snippet =
            start.as_ref().and_then(|e| self.r.tcx.sess.source_map().span_to_snippet(e.span).ok());
        let end_snippet =
            end.as_ref().and_then(|e| self.r.tcx.sess.source_map().span_to_snippet(e.span).ok());

        let field = |name: &str, val: String| {
            if val == name { val } else { format!("{name}: {val}") }
        };

        let mut resolve_short_name = |short: Symbol, full: &str| -> String {
            let ident = Ident::with_dummy_span(short);
            let path = Segment::from_path(&Path::from_ident(ident));

            match self.resolve_path(&path, Some(TypeNS), None, PathSource::Type) {
                PathResult::NonModule(..) => short.to_string(),
                _ => full.to_string(),
            }
        };
        // FIXME(new_range): Also account for new range types
        let (struct_path, fields) = match (start_snippet, end_snippet, &end_kind.node) {
            (Some(start), Some(end), ast::RangeEnd::Excluded) => (
                resolve_short_name(sym::Range, "std::ops::Range"),
                vec![field("start", start), field("end", end)],
            ),
            (Some(start), Some(end), ast::RangeEnd::Included(_)) => (
                resolve_short_name(sym::RangeInclusive, "std::ops::RangeInclusive"),
                vec![field("start", start), field("end", end)],
            ),
            (Some(start), None, _) => (
                resolve_short_name(sym::RangeFrom, "std::ops::RangeFrom"),
                vec![field("start", start)],
            ),
            (None, Some(end), ast::RangeEnd::Excluded) => {
                (resolve_short_name(sym::RangeTo, "std::ops::RangeTo"), vec![field("end", end)])
            }
            (None, Some(end), ast::RangeEnd::Included(_)) => (
                resolve_short_name(sym::RangeToInclusive, "std::ops::RangeToInclusive"),
                vec![field("end", end)],
            ),
            _ => return,
        };

        err.span_suggestion_verbose(
            pat.span,
            format!("if you meant to destructure a range use a struct pattern"),
            format!("{} {{ {} }}", struct_path, fields.join(", ")),
            Applicability::MaybeIncorrect,
        );

        err.note(
            "range patterns match against the start and end of a range; \
             to bind the components, use a struct pattern",
        );
    }

    fn suggest_swapping_misplaced_self_ty_and_trait(
        &mut self,
        err: &mut Diag<'_>,
        source: PathSource<'_, 'ast, 'ra>,
        res: Option<Res>,
        span: Span,
    ) {
        if let Some((trait_ref, self_ty)) =
            self.diag_metadata.currently_processing_impl_trait.clone()
            && let TyKind::Path(_, self_ty_path) = &self_ty.kind
            && let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(&Segment::from_path(self_ty_path), Some(TypeNS), None, source)
            && module.def_kind() == Some(DefKind::Trait)
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
        &self,
        err: &mut Diag<'_>,
        res: Option<Res>,
        source: PathSource<'_, '_, '_>,
    ) {
        let PathSource::TupleStruct(_, _) = source else { return };
        let Some(Res::Def(DefKind::Fn, _)) = res else { return };
        err.primary_message("expected a pattern, found a function call");
        err.note("function calls are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>");
    }

    fn suggest_changing_type_to_const_param(
        &self,
        err: &mut Diag<'_>,
        res: Option<Res>,
        source: PathSource<'_, '_, '_>,
        path: &[Segment],
        following_seg: Option<&Segment>,
        span: Span,
    ) {
        if let PathSource::Expr(None) = source
            && let Some(Res::Def(DefKind::TyParam, _)) = res
            && following_seg.is_none()
            && let [segment] = path
        {
            // We have something like
            // impl<T, N> From<[T; N]> for VecWrapper<T> {
            //     fn from(slice: [T; N]) -> Self {
            //         VecWrapper(slice.to_vec())
            //     }
            // }
            // where `N` is a type param but should likely have been a const param.
            let Some(item) = self.diag_metadata.current_item else { return };
            let Some(generics) = item.kind.generics() else { return };
            let Some(span) = generics.params.iter().find_map(|param| {
                // Only consider type params with no bounds.
                if param.bounds.is_empty() && param.ident.name == segment.ident.name {
                    Some(param.ident.span)
                } else {
                    None
                }
            }) else {
                return;
            };
            err.subdiagnostic(diagnostics::UnexpectedResChangeTyParamToConstParamSugg {
                before: span.shrink_to_lo(),
                after: span.shrink_to_hi(),
            });
            return;
        }
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
                if self.r.features.adt_const_params() || self.r.features.min_adt_const_params() =>
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
            err.subdiagnostic(diagnostics::UnexpectedResChangeTyToConstParamSugg {
                span: param.shrink_to_lo(),
                applicability,
            });
        }
    }

    fn suggest_pattern_match_with_let(
        &self,
        err: &mut Diag<'_>,
        source: PathSource<'_, '_, '_>,
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
        source: &PathSource<'_, 'ast, 'ra>,
        filter_fn: &impl Fn(Res) -> bool,
    ) -> Option<TypoSuggestion> {
        if let crate::PathSource::TraitItem(_, _) = source {
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(mod_path, None, None, *source)
            {
                let targets: Vec<_> = self
                    .r
                    .resolutions(module)
                    .borrow()
                    .iter()
                    .filter_map(|(key, resolution)| {
                        let resolution = resolution.borrow();
                        resolution.best_decl().map(|binding| binding.res()).and_then(|res| {
                            if filter_fn(res) {
                                Some((key.ident.name, resolution.orig_ident_span, res))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();
                if let &[(name, orig_ident_span, res)] = targets.as_slice() {
                    return Some(TypoSuggestion::single_item(name, orig_ident_span, res));
                }
            }
        }
        None
    }

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    fn restrict_assoc_type_in_where_clause(&self, span: Span, err: &mut Diag<'_>) -> bool {
        // Detect that we are actually in a `where` predicate.
        let Some(ast::WherePredicate {
            kind:
                ast::WherePredicateKind::BoundPredicate(ast::WhereBoundPredicate {
                    bounded_ty,
                    bound_generic_params,
                    bounds,
                }),
            span: where_span,
            ..
        }) = self.diag_metadata.current_where_predicate
        else {
            return false;
        };
        if !bound_generic_params.is_empty() {
            return false;
        }

        // Confirm that the target is an associated type.
        let ast::TyKind::Path(Some(qself), path) = &bounded_ty.kind else { return false };
        // use this to verify that ident is a type param.
        let Some(partial_res) = self.r.partial_res_map.get(&bounded_ty.id) else { return false };
        if !matches!(partial_res.full_res(), Some(Res::Def(DefKind::AssocTy, _))) {
            return false;
        }

        let peeled_ty = qself.ty.peel_refs();
        let ast::TyKind::Path(None, type_param_path) = &peeled_ty.kind else { return false };
        // Confirm that the `SelfTy` is a type parameter.
        let Some(partial_res) = self.r.partial_res_map.get(&peeled_ty.id) else {
            return false;
        };
        if !matches!(partial_res.full_res(), Some(Res::Def(DefKind::TyParam, _))) {
            return false;
        }
        let ([ast::PathSegment { args: None, .. }], [ast::GenericBound::Trait(poly_trait_ref)]) =
            (&type_param_path.segments[..], &bounds[..])
        else {
            return false;
        };
        let [ast::PathSegment { ident, args: None, id }] =
            &poly_trait_ref.trait_ref.path.segments[..]
        else {
            return false;
        };
        if poly_trait_ref.modifiers != ast::TraitBoundModifiers::NONE {
            return false;
        }
        if ident.span == span {
            let Some(partial_res) = self.r.partial_res_map.get(&id) else {
                return false;
            };
            if !matches!(partial_res.full_res(), Some(Res::Def(..))) {
                return false;
            }

            let Some(new_where_bound_predicate) =
                mk_where_bound_predicate(path, poly_trait_ref, &qself.ty)
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
        true
    }

    /// Check if the source is call expression and the first argument is `self`. If true,
    /// return the span of whole call and the span for all arguments expect the first one (`self`).
    fn call_has_self_arg(&self, source: PathSource<'_, '_, '_>) -> Option<(Span, Option<Span>)> {
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
        if let Some(open_brace_span) = sm.span_followed_by(span, "{") {
            // In case this could be a struct literal that needs to be surrounded
            // by parentheses, find the appropriate span.
            let close_brace_span =
                sm.span_to_next_source(open_brace_span).ok().and_then(|next_source| {
                    // Find the matching `}` accounting for nested braces.
                    let mut depth: u32 = 1;
                    let offset = next_source.char_indices().find_map(|(i, c)| {
                        match c {
                            '{' => depth += 1,
                            '}' if depth == 1 => return Some(i),
                            '}' => depth -= 1,
                            _ => {}
                        }
                        None
                    })?;
                    let start = open_brace_span.hi() + rustc_span::BytePos(offset as u32);
                    Some(open_brace_span.with_lo(start).with_hi(start + rustc_span::BytePos(1)))
                });
            let closing_brace = close_brace_span.map(|sp| span.to(sp));
            (true, closing_brace)
        } else {
            (false, None)
        }
    }

    fn update_err_for_private_tuple_struct_fields(
        &mut self,
        err: &mut Diag<'_>,
        source: &PathSource<'_, '_, '_>,
        def_id: DefId,
    ) -> Option<Vec<Span>> {
        match source {
            // e.g. `if let Enum::TupleVariant(field1, field2) = _`
            PathSource::TupleStruct(_, pattern_spans) => {
                err.primary_message(
                    "cannot match against a tuple struct which contains private fields",
                );

                // Use spans of the tuple struct pattern.
                Some(Vec::from(*pattern_spans))
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

                self.r
                    .field_idents(def_id)
                    .map(|fields| fields.iter().map(|f| f.span).collect::<Vec<_>>())
            }
            _ => None,
        }
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut Diag<'_>,
        span: Span,
        source: PathSource<'_, '_, '_>,
        path: &[Segment],
        res: Res,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);

        let path_sep = |this: &Self, err: &mut Diag<'_>, expr: &Expr, kind: DefKind| {
            const MESSAGE: &str = "use the path separator to refer to an item";

            let (lhs_span, rhs_span) = match &expr.kind {
                ExprKind::Field(base, ident) => (base.span, ident.span),
                ExprKind::MethodCall(MethodCall { receiver, span, .. }) => (receiver.span, *span),
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

        let find_span = |source: &PathSource<'_, '_, '_>, err: &mut Diag<'_>| {
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

                    // Imprecise for local structs without ctors, we don't keep fields for them.
                    let has_private_fields = match def_id.as_local() {
                        Some(def_id) => this.r.struct_ctors.get(&def_id).is_some_and(|ctor| {
                            ctor.has_private_fields(this.parent_scope.module, this.r)
                        }),
                        None => this.r.tcx.associated_item_def_ids(def_id).iter().any(|field_id| {
                            let vis = this.r.tcx.visibility(*field_id);
                            !this.r.is_accessible_from(vis, this.parent_scope.module)
                        }),
                    };
                    if !has_private_fields {
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

                            err.multipart_suggestion(
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
                Res::Def(DefKind::Macro(kinds), def_id),
                PathSource::Expr(Some(Expr {
                    kind: ExprKind::Index(..) | ExprKind::Call(..), ..
                }))
                | PathSource::Struct(_),
            ) if kinds.contains(MacroKinds::BANG) => {
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
            (Res::Def(DefKind::Macro(kinds), _), _) if kinds.contains(MacroKinds::BANG) => {
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
                if let PathSource::Expr(Some(parent)) = source
                    && let ExprKind::Field(..) | ExprKind::MethodCall(..) = parent.kind
                {
                    bad_struct_syntax_suggestion(self, err, def_id);
                    return true;
                }
                let Some(ctor) = self.r.struct_ctor(def_id) else {
                    bad_struct_syntax_suggestion(self, err, def_id);
                    return true;
                };

                // A type is re-exported and has an inaccessible constructor because it has fields
                // that are inaccessible from the reexport's scope, extend the diagnostic.
                let is_accessible = self.r.is_accessible_from(ctor.vis, self.parent_scope.module);
                if is_accessible
                    && let mod_path = &path[..path.len() - 1]
                    && let PathResult::Module(ModuleOrUniformRoot::Module(import_mod)) =
                        self.resolve_path(mod_path, Some(TypeNS), None, PathSource::Module)
                    && ctor.has_private_fields(import_mod, self.r)
                    && let Ok(import_decl) = self.r.cm().maybe_resolve_ident_in_module(
                        ModuleOrUniformRoot::Module(import_mod),
                        path.last().unwrap().ident,
                        TypeNS,
                        &self.parent_scope,
                        None,
                    )
                {
                    err.span_note(
                        import_decl.span,
                        "the type is accessed through this re-export, but the type's constructor \
                         is not visible in this import's scope due to private fields",
                    );
                    if !ctor.has_private_fields(self.parent_scope.module, self.r) {
                        err.span_suggestion_verbose(
                            span,
                            "the type can be constructed directly, because its fields are \
                             available from the current scope",
                            // Using `tcx.def_path_str` causes the compiler to hang.
                            // We don't need to handle foreign crate types because in that case you
                            // can't access the ctor either way.
                            format!(
                                "crate{}", // The method already has leading `::`.
                                self.r.tcx.def_path(def_id).to_string_no_crate_verbose(),
                            ),
                            Applicability::MachineApplicable,
                        );
                    }
                    self.update_err_for_private_tuple_struct_fields(err, &source, def_id);
                }
                if !is_expected(ctor.res) || is_accessible {
                    return true;
                }

                let field_spans =
                    self.update_err_for_private_tuple_struct_fields(err, &source, def_id);

                if let Some(spans) = field_spans
                    .filter(|spans| spans.len() > 0 && ctor.field_visibilities.len() == spans.len())
                {
                    let non_visible_spans: Vec<Span> = iter::zip(&ctor.field_visibilities, &spans)
                        .filter(|(vis, _)| {
                            !self.r.is_accessible_from(**vis, self.parent_scope.module)
                        })
                        .map(|(_, span)| *span)
                        .collect();

                    if non_visible_spans.len() > 0 {
                        if let Some(fields) = self.r.field_visibility_spans.get(&def_id) {
                            err.multipart_suggestion(
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
            (
                Res::Def(DefKind::TyAlias | DefKind::AssocTy, _),
                PathSource::TraitItem(ValueNS, PathSource::TupleStruct(whole, args)),
            ) => {
                err.note("can't use a type alias as tuple pattern");

                let mut suggestion = Vec::new();

                if let &&[first, ..] = args
                    && let &&[.., last] = args
                {
                    suggestion.extend([
                        // "0: " has to be included here so that the fix is machine applicable.
                        //
                        // If this would only add " { " and then the code below add "0: ",
                        // rustfix would crash, because end of this suggestion is the same as start
                        // of the suggestion below. Thus, we have to merge these...
                        (span.between(first), " { 0: ".to_owned()),
                        (last.between(whole.shrink_to_hi()), " }".to_owned()),
                    ]);

                    suggestion.extend(
                        args.iter()
                            .enumerate()
                            .skip(1) // See above
                            .map(|(index, &arg)| (arg.shrink_to_lo(), format!("{index}: "))),
                    )
                } else {
                    suggestion.push((span.between(whole.shrink_to_hi()), " {}".to_owned()));
                }

                err.multipart_suggestion(
                    "use struct pattern instead",
                    suggestion,
                    Applicability::MachineApplicable,
                );
            }
            (
                Res::Def(DefKind::TyAlias | DefKind::AssocTy, _),
                PathSource::TraitItem(
                    ValueNS,
                    PathSource::Expr(Some(ast::Expr {
                        span: whole,
                        kind: ast::ExprKind::Call(_, args),
                        ..
                    })),
                ),
            ) => {
                err.note("can't use a type alias as a constructor");

                let mut suggestion = Vec::new();

                if let [first, ..] = &**args
                    && let [.., last] = &**args
                {
                    suggestion.extend([
                        // "0: " has to be included here so that the fix is machine applicable.
                        //
                        // If this would only add " { " and then the code below add "0: ",
                        // rustfix would crash, because end of this suggestion is the same as start
                        // of the suggestion below. Thus, we have to merge these...
                        (span.between(first.span), " { 0: ".to_owned()),
                        (last.span.between(whole.shrink_to_hi()), " }".to_owned()),
                    ]);

                    suggestion.extend(
                        args.iter()
                            .enumerate()
                            .skip(1) // See above
                            .map(|(index, arg)| (arg.span.shrink_to_lo(), format!("{index}: "))),
                    )
                } else {
                    suggestion.push((span.between(whole.shrink_to_hi()), " {}".to_owned()));
                }

                err.multipart_suggestion(
                    "use struct expression instead",
                    suggestion,
                    Applicability::MachineApplicable,
                );
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
        args: &[Box<Expr>],
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
            .flat_map(|&i| self.r.tcx.associated_items(i).in_definition_order())
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
            format!("::{name}({})", std::iter::repeat_n("_", args).collect::<Vec<_>>().join(", "))
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
            .find(|did| find_attr!(self.r.tcx, *did, RustcDiagnosticItem(sym::Default)));
        let Some(default_trait) = default_trait else {
            return;
        };
        if self
            .r
            .extern_crate_map
            .items()
            // FIXME: This doesn't include impls like `impl Default for String`.
            .flat_map(|(_, crate_)| {
                UnordItems::new(
                    self.r.tcx.implementations_of_trait((*crate_, default_trait)).into_iter(),
                )
            })
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

        let targets = self
            .r
            .resolutions(*module)
            .borrow()
            .iter()
            .filter_map(|(key, res)| res.borrow().best_decl().map(|binding| (key, binding.res())))
            .filter(|(_, res)| match (kind, res) {
                (AssocItemKind::Const(..), Res::Def(DefKind::AssocConst { .. }, _)) => true,
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
            if let Some(node_id) = self.diag_metadata.current_self_type.and_then(extract_node_id)
                && let Some(resolution) = self.r.partial_res_map.get(&node_id)
                && let Some(Res::Def(DefKind::Struct | DefKind::Union, did)) = resolution.full_res()
                && let Some(fields) = self.r.field_idents(did)
                && let Some(field) = fields.iter().find(|id| ident.name == id.name)
            {
                // Look for a field with the same name in the current self_type.
                return Some(AssocSuggestion::Field(field.span));
            }
        }

        if let Some(items) = self.diag_metadata.current_trait_assoc_items {
            for assoc_item in items {
                if let Some(assoc_ident) = assoc_item.kind.ident()
                    && assoc_ident == ident
                {
                    return Some(match &assoc_item.kind {
                        ast::AssocItemKind::Const(..) => AssocSuggestion::AssocConst,
                        ast::AssocItemKind::Fn(ast::Fn { sig, .. }) if sig.decl.has_self() => {
                            AssocSuggestion::MethodWithSelf { called }
                        }
                        ast::AssocItemKind::Fn(..) => AssocSuggestion::AssocFn { called },
                        ast::AssocItemKind::Type(..) => AssocSuggestion::AssocType,
                        ast::AssocItemKind::Delegation(..)
                            if self
                                .r
                                .owners
                                .get(&assoc_item.id)
                                .and_then(|o| self.r.delegation_fn_sigs.get(&o.def_id))
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
        if let Some((module, _)) = self.current_trait_ref
            && let Ok(binding) = self.r.cm().maybe_resolve_ident_in_module(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                &self.parent_scope,
                None,
            )
        {
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
                    Res::Def(DefKind::AssocConst { .. }, _) => {
                        return Some(AssocSuggestion::AssocConst);
                    }
                    Res::Def(DefKind::AssocTy, _) => {
                        return Some(AssocSuggestion::AssocType);
                    }
                    _ => {}
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
                        names.push(TypoSuggestion::new(ident.name, ident.span, res));
                    }
                }

                if let RibKind::Block(Some(module)) = rib.kind {
                    self.r.add_module_candidates(
                        module.to_module(),
                        &mut names,
                        &filter_fn,
                        Some(ctxt),
                    );
                } else if let RibKind::Module(module) = rib.kind {
                    // Encountered a module item, abandon ribs and look into that module and preludes.
                    let parent_scope =
                        &ParentScope { module: module.to_module(), ..self.parent_scope };
                    self.r.add_scope_set_candidates(
                        &mut names,
                        ScopeSet::All(ns),
                        parent_scope,
                        segment.ident.span.with_ctxt(ctxt),
                        filter_fn,
                    );
                    break;
                }

                if let RibKind::MacroDefinition(def) = rib.kind
                    && def == self.r.macro_def(ctxt)
                {
                    // If an invocation of this macro created `ident`, give up on `ident`
                    // and switch to `ident`'s source from the macro definition.
                    ctxt.remove_mark();
                }
            }
        } else {
            // Search in module.
            let mod_path = &path[..path.len() - 1];
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) =
                self.resolve_path(mod_path, Some(TypeNS), None, PathSource::Type)
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
                Res::Def(DefKind::Mod, def_id) => {
                    let module = self.r.expect_module(def_id);
                    self.r
                        .resolutions(module)
                        .borrow()
                        .iter()
                        .any(|(key, _)| key.ident.name == following_seg.ident.name)
                }
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
    fn let_binding_suggestion(&self, err: &mut Diag<'_>, ident_span: Span) -> bool {
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
                    let name = name.trim_prefix('_');
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

    fn find_module(&self, def_id: DefId) -> Option<(Module<'ra>, ImportSuggestion)> {
        let mut result = None;
        let mut seen_modules = FxHashSet::default();
        let mut worklist = vec![(self.r.graph_root.to_module(), ThinVec::new(), true)];

        while let Some((in_module, path_segments, doc_visible)) = worklist.pop() {
            // abort if the module is already found
            if result.is_some() {
                break;
            }

            in_module.for_each_child(self.r, |r, ident, orig_ident_span, _, name_binding| {
                // abort if the module is already found or if name_binding is private external
                if result.is_some() || !name_binding.vis().is_visible_locally() {
                    return;
                }
                if let Some(module_def_id) = name_binding.res().module_like_def_id() {
                    // form the path
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ast::PathSegment::from_ident(ident.orig(orig_ident_span)));
                    let doc_visible = doc_visible
                        && (module_def_id.is_local() || !r.tcx.is_doc_hidden(module_def_id));
                    if module_def_id == def_id {
                        let path =
                            Path { span: name_binding.span, segments: path_segments, tokens: None };
                        result = Some((
                            r.expect_module(module_def_id),
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
                            let module = r.expect_module(module_def_id);
                            worklist.push((module, path_segments, doc_visible));
                        }
                    }
                }
            });
        }

        result
    }

    fn collect_enum_ctors(&self, def_id: DefId) -> Option<Vec<(Path, DefId, CtorKind)>> {
        self.find_module(def_id).map(|(enum_module, enum_import_suggestion)| {
            let mut variants = Vec::new();
            enum_module.for_each_child(self.r, |_, ident, orig_ident_span, _, name_binding| {
                if let Res::Def(DefKind::Ctor(CtorOf::Variant, kind), def_id) = name_binding.res() {
                    let mut segms = enum_import_suggestion.path.segments.clone();
                    segms.push(ast::PathSegment::from_ident(ident.orig(orig_ident_span)));
                    let path = Path { span: name_binding.span, segments: segms, tokens: None };
                    variants.push((path, def_id, kind));
                }
            });
            variants
        })
    }

    /// Adds a suggestion for using an enum's variant when an enum is used instead.
    fn suggest_using_enum_variant(
        &self,
        err: &mut Diag<'_>,
        source: PathSource<'_, '_, '_>,
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
                ExprKind::MethodCall(MethodCall {
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

    /// Detects missing const parameters in `impl` blocks and suggests adding them.
    ///
    /// When a const parameter is used in the self type of an `impl` but not declared
    /// in the `impl`'s own generic parameter list, this function emits a targeted
    /// diagnostic with a suggestion to add it at the correct position.
    ///
    /// Example:
    ///
    /// ```rust,ignore (suggested field is not completely correct, it should be a single suggestion)
    /// struct C<const A: u8, const X: u8, const P: u32>;
    ///
    /// impl Foo for C<A, X, P> {}
    /// //           ^ the struct `C` in `C<A, X, P>` is used as the self type
    /// //             ^ ^ ^ but A, X and P are not declared on the impl
    ///
    /// Suggested fix:
    ///
    /// impl<const A: u8, const X: u8, const P: u32> Foo for C<A, X, P> {}
    ///
    /// Current behavior (suggestions are emitted one-by-one):
    ///
    /// impl<const A: u8> Foo for C<A, X, P> {}
    /// impl<const X: u8> Foo for C<A, X, P> {}
    /// impl<const P: u32> Foo for C<A, X, P> {}
    ///
    /// Ideally the suggestion should aggregate them into a single line:
    ///
    /// impl<const A: u8, const X: u8, const P: u32> Foo for C<A, X, P> {}
    /// ```
    ///
    pub(crate) fn detect_and_suggest_const_parameter_error(
        &mut self,
        path: &[Segment],
        source: PathSource<'_, 'ast, 'ra>,
    ) -> Option<Diag<'tcx>> {
        let Some(item) = self.diag_metadata.current_item else { return None };
        let ItemKind::Impl(impl_) = &item.kind else { return None };
        let self_ty = &impl_.self_ty;

        // Represents parameter to the struct whether `A`, `X` or `P`
        let [current_parameter] = path else {
            return None;
        };

        let target_ident = current_parameter.ident;

        // Find the parent segment i.e `C` in `C<A, X, C>`
        let visitor = ParentPathVisitor::new(self_ty, target_ident);

        let Some(parent_segment) = visitor.parent else {
            return None;
        };

        let Some(args) = parent_segment.args.as_ref() else {
            return None;
        };

        let GenericArgs::AngleBracketed(angle) = args.as_ref() else {
            return None;
        };

        // Build map: NodeId of each usage in C<A, X, C> -> its position
        // e.g NodeId(A) -> 0, NodeId(X) -> 1, NodeId(C) -> 2
        let usage_to_pos: FxHashMap<NodeId, usize> = angle
            .args
            .iter()
            .enumerate()
            .filter_map(|(pos, arg)| {
                if let AngleBracketedArg::Arg(GenericArg::Type(ty)) = arg
                    && let TyKind::Path(_, path) = &ty.kind
                    && let [segment] = path.segments.as_slice()
                {
                    Some((segment.id, pos))
                } else {
                    None
                }
            })
            .collect();

        // Get the position of the missing param in C<A, X, C>
        // e.g for missing `B` in `C<A, B, C>` this gives idx=1
        let Some(idx) = current_parameter.id.and_then(|id| usage_to_pos.get(&id).copied()) else {
            return None;
        };

        // Now resolve the parent struct `C` to get its definition
        let ns = source.namespace();
        let segment = Segment::from(parent_segment);
        let segments = [segment];
        let finalize = Finalize::new(parent_segment.id, parent_segment.ident.span);

        if let Ok(Some(resolve)) = self.resolve_qpath_anywhere(
            &None,
            &segments,
            ns,
            source.defer_to_typeck(),
            finalize,
            source,
        ) && let Some(resolve) = resolve.full_res()
            && let Res::Def(_, def_id) = resolve
            && def_id.is_local()
            && let Some(local_def_id) = def_id.as_local()
            && let Some(struct_generics) = self.r.struct_generics.get(&local_def_id)
            && let Some(target_param) = &struct_generics.params.get(idx)
            && let GenericParamKind::Const { ty, .. } = &target_param.kind
            && let TyKind::Path(_, path) = &ty.kind
        {
            let full_type = path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");

            // Find the first impl param whose position in C<A, X, C>
            // is strictly greater than our missing param's index
            // e.g missing B(idx=1), impl has A(pos=0) and C(pos=2)
            // C has pos=2 > 1 so insert before C
            let next_impl_param = impl_.generics.params.iter().find(|impl_param| {
                angle
                    .args
                    .iter()
                    .find_map(|arg| {
                        if let AngleBracketedArg::Arg(GenericArg::Type(ty)) = arg
                            && let TyKind::Path(_, path) = &ty.kind
                            && let [segment] = path.segments.as_slice()
                            && segment.ident == impl_param.ident
                        {
                            usage_to_pos.get(&segment.id).copied()
                        } else {
                            None
                        }
                    })
                    .map_or(false, |pos| pos > idx)
            });

            let (insert_span, snippet) = match next_impl_param {
                Some(next_param) => {
                    // Insert in the middle before next_param
                    // e.g impl<A, C> -> impl<A, const B: u8, C>
                    (
                        next_param.span().shrink_to_lo(),
                        format!("const {}: {}, ", target_ident, full_type),
                    )
                }
                None => match impl_.generics.params.last() {
                    Some(last) => {
                        // Append after last existing param
                        // e.g impl<A, B> -> impl<A, B, const C: u8>
                        (
                            last.span().shrink_to_hi(),
                            format!(", const {}: {}", target_ident, full_type),
                        )
                    }
                    None => {
                        // No generics at all on impl
                        // e.g impl Foo for C<A> -> impl<const A: u8> Foo for C<A>
                        (
                            impl_.generics.span.shrink_to_hi(),
                            format!("<const {}: {}>", target_ident, full_type),
                        )
                    }
                },
            };

            let mut err = self.r.dcx().struct_span_err(
                target_ident.span,
                format!("cannot find const `{}` in this scope", target_ident),
            );

            err.code(E0425);

            err.span_label(target_ident.span, "not found in this scope");

            err.span_label(
                target_param.span(),
                format!("corresponding const parameter on the type defined here",),
            );

            err.subdiagnostic(diagnostics::UnexpectedMissingConstParameter {
                span: insert_span,
                snippet,
                item_name: format!("{}", target_ident),
                item_location: String::from("impl"),
            });

            return Some(err);
        }

        None
    }

    pub(crate) fn suggest_adding_generic_parameter(
        &mut self,
        path: &[Segment],
        source: PathSource<'_, 'ast, 'ra>,
    ) -> (Option<(Span, &'static str, String, Applicability)>, Option<Diag<'tcx>>) {
        let (ident, span) = match path {
            [segment]
                if !segment.has_generic_args
                    && segment.ident.name != kw::SelfUpper
                    && segment.ident.name != kw::Dyn =>
            {
                (segment.ident.to_string(), segment.ident.span)
            }
            _ => return (None, None),
        };
        let mut iter = ident.chars().map(|c| c.is_uppercase());
        let single_uppercase_char =
            matches!(iter.next(), Some(true)) && matches!(iter.next(), None);
        if !self.diag_metadata.currently_processing_generic_args && !single_uppercase_char {
            return (None, None);
        }
        match (
            self.diag_metadata.current_item,
            single_uppercase_char,
            self.diag_metadata.currently_processing_generic_args,
        ) {
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
                true,
                _,
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
                        return (None, None);
                    }

                    let (msg, sugg) = match source {
                        PathSource::Type | PathSource::PreciseCapturingArg(TypeNS) => {
                            if let Some(err) =
                                self.detect_and_suggest_const_parameter_error(path, source)
                            {
                                return (None, Some(err));
                            }
                            ("you might be missing a type parameter", ident)
                        }
                        PathSource::Expr(_) | PathSource::PreciseCapturingArg(ValueNS) => (
                            "you might be missing a const parameter",
                            format!("const {ident}: /* Type */"),
                        ),
                        _ => return (None, None),
                    };
                    let (span, sugg) = if let [.., param] = &generics.params[..] {
                        let span = if let [.., bound] = &param.bounds[..] {
                            bound.span()
                        } else if let GenericParam {
                            kind: GenericParamKind::Const { ty, span: _, default },
                            ..
                        } = param
                        {
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
                        return (
                            Some((span.shrink_to_hi(), msg, sugg, Applicability::MaybeIncorrect)),
                            None,
                        );
                    }
                }
            }
            _ => {}
        }
        (None, None)
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
                // A lifetime bound is a real use of that lifetime parameter, even
                // though visiting a bound like `'b: 'a` only records a use of `'a`.
                Some(LifetimeUseSet::One { .. }) if !param.bounds.is_empty() => {}
                Some(LifetimeUseSet::One { use_span, use_ctxt }) => {
                    let param_ident = param.ident;
                    let deletion_span =
                        if param.bounds.is_empty() { deletion_span() } else { None };
                    self.r.lint_buffer.dyn_buffer_lint_any(
                        lint::builtin::SINGLE_USE_LIFETIMES,
                        param.id,
                        param_ident.span,
                        move |dcx, level, sess| {
                            debug!(?param_ident, ?param_ident.span, ?use_span);

                            let elidable = matches!(use_ctxt, LifetimeCtxt::Ref);
                            let suggestion = if let Some(deletion_span) = deletion_span {
                                let (use_span, replace_lt) = if elidable {
                                    let use_span = sess
                                        .downcast_ref::<Session>()
                                        .expect("expected a `Session`")
                                        .source_map()
                                        .span_extend_while_whitespace(use_span);
                                    (use_span, String::new())
                                } else {
                                    (use_span, "'_".to_owned())
                                };
                                debug!(?deletion_span, ?use_span);

                                // issue 107998 for the case such as a wrong function pointer type
                                // `deletion_span` is empty and there is no need to report lifetime uses here
                                let deletion_span = if deletion_span.is_empty() {
                                    None
                                } else {
                                    Some(deletion_span)
                                };
                                Some(diagnostics::SingleUseLifetimeSugg {
                                    deletion_span,
                                    use_span,
                                    replace_lt,
                                })
                            } else {
                                None
                            };
                            diagnostics::SingleUseLifetime {
                                suggestion,
                                param_span: param_ident.span,
                                use_span,
                                ident: param_ident,
                            }
                            .into_diag(dcx, level)
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
                            diagnostics::UnusedLifetime { deletion_span, ident: param.ident },
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
    ) -> ErrorGuaranteed {
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
                Some(lifetime_ref.ident),
                |err, _, span, message, suggestion, span_suggs| {
                    err.multipart_suggestion(
                        message,
                        std::iter::once((span, suggestion)).chain(span_suggs).collect(),
                        Applicability::MaybeIncorrect,
                    );
                    true
                },
            );
        }

        err.emit()
    }

    fn suggest_introducing_lifetime(
        &self,
        err: &mut Diag<'_>,
        name: Option<Ident>,
        suggest: impl Fn(
            &mut Diag<'_>,
            bool,
            Span,
            Cow<'static, str>,
            String,
            Vec<(Span, String)>,
        ) -> bool,
    ) {
        self.suggest_introducing_lifetime_filtered(err, name, |_| true, suggest);
    }

    pub(crate) fn suggest_introducing_lifetime_for_assoc_ty_binding(
        &self,
        err: &mut Diag<'_>,
        lifetime: Span,
    ) {
        self.suggest_introducing_lifetime_filtered(
            err,
            None,
            |kind| {
                !matches!(
                    kind,
                    LifetimeBinderKind::FnPtrType
                        | LifetimeBinderKind::PolyTrait
                        | LifetimeBinderKind::WhereBound
                )
            },
            |err, _higher_ranked, span, message, intro_sugg, _| {
                err.multipart_suggestion(
                    message,
                    vec![(span, intro_sugg), (lifetime.shrink_to_hi(), "'a ".to_string())],
                    Applicability::MaybeIncorrect,
                );
                false
            },
        );
    }

    fn suggest_introducing_lifetime_filtered(
        &self,
        err: &mut Diag<'_>,
        name: Option<Ident>,
        mut consider: impl FnMut(LifetimeBinderKind) -> bool,
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
                    if matches!(kind, LifetimeBinderKind::ImplAssocType) || !consider(kind) {
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
                        LifetimeBinderKind::FnPtrType
                            | LifetimeBinderKind::PolyTrait
                            | LifetimeBinderKind::WhereBound
                    );

                    let mut rm_inner_binders: FxIndexSet<Span> = Default::default();
                    let (span, sugg) = if span.is_empty() {
                        let mut binder_idents: FxIndexSet<Ident> = Default::default();
                        binder_idents.insert(name.unwrap_or(Ident::from_str("'a")));

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

                        let binders_sugg: String = std::iter::Iterator::intersperse(
                            binder_idents.into_iter().map(|ident| ident.to_string()),
                            ", ".to_owned(),
                        )
                        .collect();
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
                        let sugg =
                            format!("{}, ", name.map(|i| i.to_string()).as_deref().unwrap_or("'a"));
                        (span, sugg)
                    };

                    if higher_ranked {
                        let message = Cow::from(format!(
                            "consider making the {} lifetime-generic with a new `{}` lifetime",
                            kind.descr(),
                            name.map(|i| i.to_string()).as_deref().unwrap_or("'a"),
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

    pub(crate) fn emit_non_static_lt_in_const_param_ty_error(
        &self,
        lifetime_ref: &ast::Lifetime,
    ) -> ErrorGuaranteed {
        self.r
            .dcx()
            .create_err(diagnostics::ParamInTyOfConstParam {
                span: lifetime_ref.ident.span,
                name: lifetime_ref.ident.name,
            })
            .emit()
    }

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `generic_const_exprs` is not enabled, the body identified by
    /// `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    pub(crate) fn emit_forbidden_non_static_lifetime_error(
        &self,
        cause: NoConstantGenericsReason,
        lifetime_ref: &ast::Lifetime,
    ) -> ErrorGuaranteed {
        match cause {
            NoConstantGenericsReason::IsEnumDiscriminant => self
                .r
                .dcx()
                .create_err(diagnostics::ParamInEnumDiscriminant {
                    span: lifetime_ref.ident.span,
                    name: lifetime_ref.ident.name,
                    param_kind: diagnostics::ParamKindInEnumDiscriminant::Lifetime,
                })
                .emit(),
            NoConstantGenericsReason::NonTrivialConstArg => {
                assert!(!self.r.features.generic_const_exprs());
                self.r
                    .dcx()
                    .create_err(diagnostics::ParamInNonTrivialAnonConst {
                        span: lifetime_ref.ident.span,
                        name: lifetime_ref.ident.name,
                        param_kind: diagnostics::ParamKindInNonTrivialAnonConst::Lifetime,
                        help: self.r.tcx.sess.is_nightly_build(),
                        is_gca: self.r.features.generic_const_args(),
                        help_gca: self.r.features.generic_const_args(),
                    })
                    .emit()
            }
        }
    }

    pub(crate) fn report_missing_lifetime_specifiers<'a>(
        &mut self,
        lifetime_refs: impl Clone + IntoIterator<Item = &'a MissingLifetime>,
        function_param_lifetimes: Option<(Vec<MissingLifetime>, Vec<ElisionFnParameter>)>,
    ) -> ErrorGuaranteed {
        let num_lifetimes: usize = lifetime_refs.clone().into_iter().map(|lt| lt.count).sum();
        let spans: Vec<_> = lifetime_refs.clone().into_iter().map(|lt| lt.span).collect();

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

    fn add_missing_lifetime_specifiers_label<'a>(
        &mut self,
        err: &mut Diag<'_>,
        lifetime_refs: impl Clone + IntoIterator<Item = &'a MissingLifetime>,
        function_param_lifetimes: Option<(Vec<MissingLifetime>, Vec<ElisionFnParameter>)>,
    ) {
        for &lt in lifetime_refs.clone() {
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
                        (DUMMY_NODE_ID, LifetimeRes::Static),
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
                        (DUMMY_NODE_ID, LifetimeRes::Static),
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
        let source_map = self.r.tcx.sess.source_map();
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
                let sugg: String = std::iter::Iterator::intersperse(
                    std::iter::repeat_n(existing_name.as_str(), lt.count),
                    ", ",
                )
                .collect();
                let is_empty_brackets = source_map.span_followed_by(lt.span, ">").is_some();
                let sugg = if is_empty_brackets { sugg } else { format!("{sugg}, ") };
                (lt.span.shrink_to_hi(), sugg)
            }
            MissingLifetimeKind::Brackets => {
                let sugg: String = std::iter::once("<")
                    .chain(std::iter::Iterator::intersperse(
                        std::iter::repeat_n(existing_name.as_str(), lt.count),
                        ", ",
                    ))
                    .chain([">"])
                    .collect();
                (lt.span.shrink_to_hi(), sugg)
            }
        };
        for &lt in lifetime_refs.clone() {
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
                        err.multipart_suggestion(
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
                    let mut lifetime_refs = lifetime_refs.clone().into_iter();
                    let owned = if let Some(lt) = lifetime_refs.next()
                        && lifetime_refs.next().is_none()
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
                err.multipart_suggestion(
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
                    let mut lifetime_refs = lifetime_refs.clone().into_iter();
                    if let Some(lt) = lifetime_refs.next()
                        && lifetime_refs.next().is_none()
                        && (lt.kind == MissingLifetimeKind::Ampersand
                            || lt.kind == MissingLifetimeKind::Underscore)
                    {
                        let pre = if let Some((kind, _span)) = self.diag_metadata.current_function
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
                            let dotdotdot =
                                if lt.kind == MissingLifetimeKind::Ampersand { "..." } else { "" };
                            err.multipart_suggestion(
                                format!(
                                    "instead, you are more likely to want to change {the} \
                                     argument{s} to be borrowed{dotdotdot}",
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
                                    err.multipart_suggestion(
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
                        let mut sugg_is_str_to_string = false;
                        let mut sugg = vec![(lt.span, String::new())];
                        if let Some((kind, _span)) = self.diag_metadata.current_function
                            && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = kind
                        {
                            let mut lt_finder =
                                LifetimeFinder { lifetime: lt.span, found: None, seen: vec![] };
                            for param in &sig.decl.inputs {
                                lt_finder.visit_ty(&param.ty);
                            }
                            if let ast::FnRetTy::Ty(ret_ty) = &sig.decl.output {
                                lt_finder.visit_ty(ret_ty);
                                let mut ret_lt_finder =
                                    LifetimeFinder { lifetime: lt.span, found: None, seen: vec![] };
                                ret_lt_finder.visit_ty(ret_ty);
                                if let [Ty { span, kind: TyKind::Ref(_, mut_ty), .. }] =
                                    &ret_lt_finder.seen[..]
                                {
                                    // We might have a situation like
                                    // fn g(mut x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()>
                                    // but `lt.span` only points at `'_`, so to suggest `-> Option<()>`
                                    // we need to find a more accurate span to end up with
                                    // fn g<'a>(mut x: impl Iterator<Item = &'_ ()>) -> Option<()>
                                    sugg = vec![(span.with_hi(mut_ty.ty.span.lo()), String::new())];
                                    owned_sugg = true;
                                }
                            }
                            if let Some(ty) = lt_finder.found {
                                if let TyKind::Path(None, path) = &ty.kind {
                                    // Check if the path being borrowed is likely to be owned.
                                    let path: Vec<_> = Segment::from_path(path);
                                    match self.resolve_path(
                                        &path,
                                        Some(TypeNS),
                                        None,
                                        PathSource::Type,
                                    ) {
                                        PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                                            match module.res() {
                                                Some(Res::PrimTy(PrimTy::Str)) => {
                                                    // Don't suggest `-> str`, suggest `-> String`.
                                                    sugg = vec![(
                                                        lt.span.with_hi(ty.span.hi()),
                                                        "String".to_string(),
                                                    )];
                                                    sugg_is_str_to_string = true;
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
                                                    sugg_is_str_to_string = true;
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
                            if let Some(span) =
                                self.find_ref_prefix_span_for_owned_suggestion(lt.span)
                                && !sugg_is_str_to_string
                            {
                                sugg = vec![(span, String::new())];
                            }
                            err.multipart_suggestion(
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
                    err.multipart_suggestion(
                        "consider using one of the available lifetimes here",
                        spans_suggs,
                        Applicability::HasPlaceholders,
                    );
                }
            }
        }
    }

    fn find_ref_prefix_span_for_owned_suggestion(&self, lifetime: Span) -> Option<Span> {
        let mut finder = RefPrefixSpanFinder { lifetime, span: None };
        if let Some(item) = self.diag_metadata.current_item {
            finder.visit_item(item);
        } else if let Some((kind, _span)) = self.diag_metadata.current_function
            && let FnKind::Fn(_, _, ast::Fn { sig, .. }) = kind
        {
            for param in &sig.decl.inputs {
                finder.visit_ty(&param.ty);
            }
            if let ast::FnRetTy::Ty(ret_ty) = &sig.decl.output {
                finder.visit_ty(ret_ty);
            }
        }
        finder.span
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
                term: ast::Term::Ty(Box::new(ast::Ty {
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
                    Some(Box::new(ast::GenericArgs::AngleBracketed(ast::AngleBracketedArgs {
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
        bounded_ty: Box::new(ty.clone()),
        bounds: thin_vec![ast::GenericBound::Trait(ast::PolyTraitRef {
            bound_generic_params: ThinVec::new(),
            modifiers: ast::TraitBoundModifiers::NONE,
            trait_ref: ast::TraitRef {
                path: ast::Path { segments: modified_segments, span: DUMMY_SP, tokens: None },
                ref_id: DUMMY_NODE_ID,
            },
            span: DUMMY_SP,
            parens: ast::Parens::No,
        })],
    };

    Some(new_where_bound_predicate)
}

/// Report lifetime/lifetime shadowing as an error.
pub(crate) fn signal_lifetime_shadowing(
    sess: &Session,
    orig: Ident,
    shadower: Ident,
) -> ErrorGuaranteed {
    struct_span_code_err!(
        sess.dcx(),
        shadower.span,
        E0496,
        "lifetime name `{}` shadows a lifetime name that is already in scope",
        orig.name,
    )
    .with_span_label(orig.span, "first declared here")
    .with_span_label(shadower.span, format!("lifetime `{}` already in scope", orig.name))
    .emit()
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

struct RefPrefixSpanFinder {
    lifetime: Span,
    span: Option<Span>,
}

impl<'ast> Visitor<'ast> for RefPrefixSpanFinder {
    fn visit_ty(&mut self, t: &'ast Ty) {
        if self.span.is_some() {
            return;
        }
        if let TyKind::Ref(_, mut_ty) | TyKind::PinnedRef(_, mut_ty) = &t.kind
            && t.span.lo() == self.lifetime.lo()
        {
            self.span = Some(t.span.with_hi(mut_ty.ty.span.lo()));
            return;
        }
        walk_ty(self, t);
    }
}

/// Shadowing involving a label is only a warning for historical reasons.
//FIXME: make this a proper lint.
pub(crate) fn signal_label_shadowing(sess: &Session, orig: Span, shadower: Ident) {
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

struct ParentPathVisitor<'a> {
    target: Ident,
    parent: Option<&'a PathSegment>,
    stack: Vec<&'a Ty>,
}

impl<'a> ParentPathVisitor<'a> {
    fn new(self_ty: &'a Ty, target: Ident) -> Self {
        let mut v = ParentPathVisitor { target, parent: None, stack: Vec::new() };

        v.visit_ty(self_ty);
        v
    }
}

impl<'a> Visitor<'a> for ParentPathVisitor<'a> {
    fn visit_ty(&mut self, ty: &'a Ty) {
        if self.parent.is_some() {
            return;
        }

        // push current type
        self.stack.push(ty);

        if let TyKind::Path(_, path) = &ty.kind
            // is this just `N`?
            && let [segment] = path.segments.as_slice()
            && segment.ident == self.target
            // parent is previous element in stack
            && let [.., parent_ty, _ty] = self.stack.as_slice()
            && let TyKind::Path(_, parent_path) = &parent_ty.kind
        {
            self.parent = parent_path.segments.first();
        }

        walk_ty(self, ty);

        self.stack.pop();
    }
}
