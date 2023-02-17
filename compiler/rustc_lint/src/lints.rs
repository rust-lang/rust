#![allow(rustc::untranslatable_diagnostic)]
#![allow(rustc::diagnostic_outside_of_impl)]
use std::num::NonZeroU32;

use rustc_errors::{
    fluent, AddToDiagnostic, Applicability, DecorateLint, DiagnosticMessage,
    DiagnosticStyledString, SuggestionStyle,
};
use rustc_hir::def_id::DefId;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{PolyExistentialTraitRef, Predicate, Ty, TyCtxt};
use rustc_session::parse::ParseSess;
use rustc_span::{edition::Edition, sym, symbol::Ident, Span, Symbol};

use crate::{
    builtin::InitError, builtin::TypeAliasBounds, errors::OverruledAttributeSub, LateContext,
};

// array_into_iter.rs
#[derive(LintDiagnostic)]
#[diag(lint_array_into_iter)]
pub struct ArrayIntoIterDiag<'a> {
    pub target: &'a str,
    #[suggestion(use_iter_suggestion, code = "iter", applicability = "machine-applicable")]
    pub suggestion: Span,
    #[subdiagnostic]
    pub sub: Option<ArrayIntoIterDiagSub>,
}

#[derive(Subdiagnostic)]
pub enum ArrayIntoIterDiagSub {
    #[suggestion(remove_into_iter_suggestion, code = "", applicability = "maybe-incorrect")]
    RemoveIntoIter {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(use_explicit_into_iter_suggestion, applicability = "maybe-incorrect")]
    UseExplicitIntoIter {
        #[suggestion_part(code = "IntoIterator::into_iter(")]
        start_span: Span,
        #[suggestion_part(code = ")")]
        end_span: Span,
    },
}

// builtin.rs
#[derive(LintDiagnostic)]
#[diag(lint_builtin_while_true)]
pub struct BuiltinWhileTrue {
    #[suggestion(style = "short", code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_box_pointers)]
pub struct BuiltinBoxPointers<'a> {
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_non_shorthand_field_patterns)]
pub struct BuiltinNonShorthandFieldPatterns {
    pub ident: Ident,
    #[suggestion(code = "{prefix}{ident}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub prefix: &'static str,
}

#[derive(LintDiagnostic)]
pub enum BuiltinUnsafe {
    #[diag(lint_builtin_allow_internal_unsafe)]
    AllowInternalUnsafe,
    #[diag(lint_builtin_unsafe_block)]
    UnsafeBlock,
    #[diag(lint_builtin_unsafe_trait)]
    UnsafeTrait,
    #[diag(lint_builtin_unsafe_impl)]
    UnsafeImpl,
    #[diag(lint_builtin_no_mangle_fn)]
    #[note(lint_builtin_overridden_symbol_name)]
    NoMangleFn,
    #[diag(lint_builtin_export_name_fn)]
    #[note(lint_builtin_overridden_symbol_name)]
    ExportNameFn,
    #[diag(lint_builtin_link_section_fn)]
    #[note(lint_builtin_overridden_symbol_section)]
    LinkSectionFn,
    #[diag(lint_builtin_no_mangle_static)]
    #[note(lint_builtin_overridden_symbol_name)]
    NoMangleStatic,
    #[diag(lint_builtin_export_name_static)]
    #[note(lint_builtin_overridden_symbol_name)]
    ExportNameStatic,
    #[diag(lint_builtin_link_section_static)]
    #[note(lint_builtin_overridden_symbol_section)]
    LinkSectionStatic,
    #[diag(lint_builtin_no_mangle_method)]
    #[note(lint_builtin_overridden_symbol_name)]
    NoMangleMethod,
    #[diag(lint_builtin_export_name_method)]
    #[note(lint_builtin_overridden_symbol_name)]
    ExportNameMethod,
    #[diag(lint_builtin_decl_unsafe_fn)]
    DeclUnsafeFn,
    #[diag(lint_builtin_decl_unsafe_method)]
    DeclUnsafeMethod,
    #[diag(lint_builtin_impl_unsafe_method)]
    ImplUnsafeMethod,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_missing_doc)]
pub struct BuiltinMissingDoc<'a> {
    pub article: &'a str,
    pub desc: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_missing_copy_impl)]
pub struct BuiltinMissingCopyImpl;

pub struct BuiltinMissingDebugImpl<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> DecorateLint<'a, ()> for BuiltinMissingDebugImpl<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("debug", self.tcx.def_path_str(self.def_id));
        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        fluent::lint_builtin_missing_debug_impl
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_anonymous_params)]
pub struct BuiltinAnonymousParams<'a> {
    #[suggestion(code = "_: {ty_snip}")]
    pub suggestion: (Span, Applicability),
    pub ty_snip: &'a str,
}

// FIXME(davidtwco) translatable deprecated attr
#[derive(LintDiagnostic)]
#[diag(lint_builtin_deprecated_attr_link)]
pub struct BuiltinDeprecatedAttrLink<'a> {
    pub name: Symbol,
    pub reason: &'a str,
    pub link: &'a str,
    #[subdiagnostic]
    pub suggestion: BuiltinDeprecatedAttrLinkSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub enum BuiltinDeprecatedAttrLinkSuggestion<'a> {
    #[suggestion(msg_suggestion, code = "", applicability = "machine-applicable")]
    Msg {
        #[primary_span]
        suggestion: Span,
        msg: &'a str,
    },
    #[suggestion(default_suggestion, code = "", applicability = "machine-applicable")]
    Default {
        #[primary_span]
        suggestion: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_deprecated_attr_used)]
pub struct BuiltinDeprecatedAttrUsed {
    pub name: String,
    #[suggestion(
        lint_builtin_deprecated_attr_default_suggestion,
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unused_doc_comment)]
pub struct BuiltinUnusedDocComment<'a> {
    pub kind: &'a str,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sub: BuiltinUnusedDocCommentSub,
}

#[derive(Subdiagnostic)]
pub enum BuiltinUnusedDocCommentSub {
    #[help(plain_help)]
    PlainHelp,
    #[help(block_help)]
    BlockHelp,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_no_mangle_generic)]
pub struct BuiltinNoMangleGeneric {
    // Use of `#[no_mangle]` suggests FFI intent; correct
    // fix may be to monomorphize source by hand
    #[suggestion(style = "short", code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_const_no_mangle)]
pub struct BuiltinConstNoMangle {
    #[suggestion(code = "pub static", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_mutable_transmutes)]
pub struct BuiltinMutablesTransmutes;

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unstable_features)]
pub struct BuiltinUnstableFeatures;

// lint_ungated_async_fn_track_caller
pub struct BuiltinUngatedAsyncFnTrackCaller<'a> {
    pub label: Span,
    pub parse_sess: &'a ParseSess,
}

impl<'a> DecorateLint<'a, ()> for BuiltinUngatedAsyncFnTrackCaller<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.span_label(self.label, fluent::label);
        rustc_session::parse::add_feature_diagnostics(
            diag,
            &self.parse_sess,
            sym::closure_track_caller,
        );
        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        fluent::lint_ungated_async_fn_track_caller
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unreachable_pub)]
pub struct BuiltinUnreachablePub<'a> {
    pub what: &'a str,
    #[suggestion(code = "pub(crate)")]
    pub suggestion: (Span, Applicability),
    #[help]
    pub help: Option<()>,
}

pub struct SuggestChangingAssocTypes<'a, 'b> {
    pub ty: &'a rustc_hir::Ty<'b>,
}

impl AddToDiagnostic for SuggestChangingAssocTypes<'_, '_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        // Access to associates types should use `<T as Bound>::Assoc`, which does not need a
        // bound. Let's see if this type does that.

        // We use a HIR visitor to walk the type.
        use rustc_hir::intravisit::{self, Visitor};
        struct WalkAssocTypes<'a> {
            err: &'a mut rustc_errors::Diagnostic,
        }
        impl Visitor<'_> for WalkAssocTypes<'_> {
            fn visit_qpath(
                &mut self,
                qpath: &rustc_hir::QPath<'_>,
                id: rustc_hir::HirId,
                span: Span,
            ) {
                if TypeAliasBounds::is_type_variable_assoc(qpath) {
                    self.err.span_help(span, fluent::lint_builtin_type_alias_bounds_help);
                }
                intravisit::walk_qpath(self, qpath, id)
            }
        }

        // Let's go for a walk!
        let mut visitor = WalkAssocTypes { err: diag };
        visitor.visit_ty(self.ty);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_type_alias_where_clause)]
pub struct BuiltinTypeAliasWhereClause<'a, 'b> {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub suggestion: Span,
    #[subdiagnostic]
    pub sub: Option<SuggestChangingAssocTypes<'a, 'b>>,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_type_alias_generic_bounds)]
pub struct BuiltinTypeAliasGenericBounds<'a, 'b> {
    #[subdiagnostic]
    pub suggestion: BuiltinTypeAliasGenericBoundsSuggestion,
    #[subdiagnostic]
    pub sub: Option<SuggestChangingAssocTypes<'a, 'b>>,
}

pub struct BuiltinTypeAliasGenericBoundsSuggestion {
    pub suggestions: Vec<(Span, String)>,
}

impl AddToDiagnostic for BuiltinTypeAliasGenericBoundsSuggestion {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.multipart_suggestion(
            fluent::suggestion,
            self.suggestions,
            Applicability::MachineApplicable,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_trivial_bounds)]
pub struct BuiltinTrivialBounds<'a> {
    pub predicate_kind_name: &'a str,
    pub predicate: Predicate<'a>,
}

#[derive(LintDiagnostic)]
pub enum BuiltinEllipsisInclusiveRangePatternsLint {
    #[diag(lint_builtin_ellipsis_inclusive_range_patterns)]
    Parenthesise {
        #[suggestion(code = "{replace}", applicability = "machine-applicable")]
        suggestion: Span,
        replace: String,
    },
    #[diag(lint_builtin_ellipsis_inclusive_range_patterns)]
    NonParenthesise {
        #[suggestion(style = "short", code = "..=", applicability = "machine-applicable")]
        suggestion: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unnameable_test_items)]
pub struct BuiltinUnnameableTestItems;

#[derive(LintDiagnostic)]
#[diag(lint_builtin_keyword_idents)]
pub struct BuiltinKeywordIdents {
    pub kw: Ident,
    pub next: Edition,
    #[suggestion(code = "r#{kw}", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_explicit_outlives)]
pub struct BuiltinExplicitOutlives {
    pub count: usize,
    #[subdiagnostic]
    pub suggestion: BuiltinExplicitOutlivesSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion)]
pub struct BuiltinExplicitOutlivesSuggestion {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
    #[applicability]
    pub applicability: Applicability,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_incomplete_features)]
pub struct BuiltinIncompleteFeatures {
    pub name: Symbol,
    #[subdiagnostic]
    pub note: Option<BuiltinIncompleteFeaturesNote>,
    #[subdiagnostic]
    pub help: Option<BuiltinIncompleteFeaturesHelp>,
}

#[derive(Subdiagnostic)]
#[help(help)]
pub struct BuiltinIncompleteFeaturesHelp;

#[derive(Subdiagnostic)]
#[note(note)]
pub struct BuiltinIncompleteFeaturesNote {
    pub n: NonZeroU32,
}

pub struct BuiltinUnpermittedTypeInit<'a> {
    pub msg: DiagnosticMessage,
    pub ty: Ty<'a>,
    pub label: Span,
    pub sub: BuiltinUnpermittedTypeInitSub,
}

impl<'a> DecorateLint<'a, ()> for BuiltinUnpermittedTypeInit<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("ty", self.ty);
        diag.span_label(self.label, fluent::lint_builtin_unpermitted_type_init_label);
        diag.span_label(self.label, fluent::lint_builtin_unpermitted_type_init_label_suggestion);
        self.sub.add_to_diagnostic(diag);
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        self.msg.clone()
    }
}

// FIXME(davidtwco): make translatable
pub struct BuiltinUnpermittedTypeInitSub {
    pub err: InitError,
}

impl AddToDiagnostic for BuiltinUnpermittedTypeInitSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        let mut err = self.err;
        loop {
            if let Some(span) = err.span {
                diag.span_note(span, err.message);
            } else {
                diag.note(err.message);
            }
            if let Some(e) = err.nested {
                err = *e;
            } else {
                break;
            }
        }
    }
}

#[derive(LintDiagnostic)]
pub enum BuiltinClashingExtern<'a> {
    #[diag(lint_builtin_clashing_extern_same_name)]
    SameName {
        this: Symbol,
        orig: Symbol,
        #[label(previous_decl_label)]
        previous_decl_label: Span,
        #[label(mismatch_label)]
        mismatch_label: Span,
        #[subdiagnostic]
        sub: BuiltinClashingExternSub<'a>,
    },
    #[diag(lint_builtin_clashing_extern_diff_name)]
    DiffName {
        this: Symbol,
        orig: Symbol,
        #[label(previous_decl_label)]
        previous_decl_label: Span,
        #[label(mismatch_label)]
        mismatch_label: Span,
        #[subdiagnostic]
        sub: BuiltinClashingExternSub<'a>,
    },
}

// FIXME(davidtwco): translatable expected/found
pub struct BuiltinClashingExternSub<'a> {
    pub tcx: TyCtxt<'a>,
    pub expected: Ty<'a>,
    pub found: Ty<'a>,
}

impl AddToDiagnostic for BuiltinClashingExternSub<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        let mut expected_str = DiagnosticStyledString::new();
        expected_str.push(self.expected.fn_sig(self.tcx).to_string(), false);
        let mut found_str = DiagnosticStyledString::new();
        found_str.push(self.found.fn_sig(self.tcx).to_string(), true);
        diag.note_expected_found(&"", expected_str, &"", found_str);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_deref_nullptr)]
pub struct BuiltinDerefNullptr {
    #[label]
    pub label: Span,
}

// FIXME: migrate fluent::lint::builtin_asm_labels

#[derive(LintDiagnostic)]
pub enum BuiltinSpecialModuleNameUsed {
    #[diag(lint_builtin_special_module_name_used_lib)]
    #[note]
    #[help]
    Lib,
    #[diag(lint_builtin_special_module_name_used_main)]
    #[note]
    Main,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unexpected_cli_config_name)]
#[help]
pub struct BuiltinUnexpectedCliConfigName {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unexpected_cli_config_value)]
#[help]
pub struct BuiltinUnexpectedCliConfigValue {
    pub name: Symbol,
    pub value: Symbol,
}

// deref_into_dyn_supertrait.rs
#[derive(LintDiagnostic)]
#[diag(lint_supertrait_as_deref_target)]
pub struct SupertraitAsDerefTarget<'a> {
    pub t: Ty<'a>,
    pub target_principal: PolyExistentialTraitRef<'a>,
    #[subdiagnostic]
    pub label: Option<SupertraitAsDerefTargetLabel>,
}

#[derive(Subdiagnostic)]
#[label(label)]
pub struct SupertraitAsDerefTargetLabel {
    #[primary_span]
    pub label: Span,
}

// enum_intrinsics_non_enums.rs
#[derive(LintDiagnostic)]
#[diag(lint_enum_intrinsics_mem_discriminant)]
pub struct EnumIntrinsicsMemDiscriminate<'a> {
    pub ty_param: Ty<'a>,
    #[note]
    pub note: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_enum_intrinsics_mem_variant)]
#[note]
pub struct EnumIntrinsicsMemVariant<'a> {
    pub ty_param: Ty<'a>,
}

// expect.rs
#[derive(LintDiagnostic)]
#[diag(lint_expectation)]
pub struct Expectation {
    #[subdiagnostic]
    pub rationale: Option<ExpectationNote>,
    #[note]
    pub note: Option<()>,
}

#[derive(Subdiagnostic)]
#[note(rationale)]
pub struct ExpectationNote {
    pub rationale: Symbol,
}

// for_loops_over_fallibles.rs
#[derive(LintDiagnostic)]
#[diag(lint_for_loops_over_fallibles)]
pub struct ForLoopsOverFalliblesDiag<'a> {
    pub article: &'static str,
    pub ty: &'static str,
    #[subdiagnostic]
    pub sub: ForLoopsOverFalliblesLoopSub<'a>,
    #[subdiagnostic]
    pub question_mark: Option<ForLoopsOverFalliblesQuestionMark>,
    #[subdiagnostic]
    pub suggestion: ForLoopsOverFalliblesSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub enum ForLoopsOverFalliblesLoopSub<'a> {
    #[suggestion(remove_next, code = ".by_ref()", applicability = "maybe-incorrect")]
    RemoveNext {
        #[primary_span]
        suggestion: Span,
        recv_snip: String,
    },
    #[multipart_suggestion(use_while_let, applicability = "maybe-incorrect")]
    UseWhileLet {
        #[suggestion_part(code = "while let {var}(")]
        start_span: Span,
        #[suggestion_part(code = ") = ")]
        end_span: Span,
        var: &'a str,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(use_question_mark, code = "?", applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesQuestionMark {
    #[primary_span]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion, applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesSuggestion<'a> {
    pub var: &'a str,
    #[suggestion_part(code = "if let {var}(")]
    pub start_span: Span,
    #[suggestion_part(code = ") = ")]
    pub end_span: Span,
}

// hidden_unicode_codepoints.rs
#[derive(LintDiagnostic)]
#[diag(lint_hidden_unicode_codepoints)]
#[note]
pub struct HiddenUnicodeCodepointsDiag<'a> {
    pub label: &'a str,
    pub count: usize,
    #[label]
    pub span_label: Span,
    #[subdiagnostic]
    pub labels: Option<HiddenUnicodeCodepointsDiagLabels>,
    #[subdiagnostic]
    pub sub: HiddenUnicodeCodepointsDiagSub,
}

pub struct HiddenUnicodeCodepointsDiagLabels {
    pub spans: Vec<(char, Span)>,
}

impl AddToDiagnostic for HiddenUnicodeCodepointsDiagLabels {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        for (c, span) in self.spans {
            diag.span_label(span, format!("{:?}", c));
        }
    }
}

pub enum HiddenUnicodeCodepointsDiagSub {
    Escape { spans: Vec<(char, Span)> },
    NoEscape { spans: Vec<(char, Span)> },
}

// Used because of multiple multipart_suggestion and note
impl AddToDiagnostic for HiddenUnicodeCodepointsDiagSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        match self {
            HiddenUnicodeCodepointsDiagSub::Escape { spans } => {
                diag.multipart_suggestion_with_style(
                    fluent::suggestion_remove,
                    spans.iter().map(|(_, span)| (*span, "".to_string())).collect(),
                    Applicability::MachineApplicable,
                    SuggestionStyle::HideCodeAlways,
                );
                diag.multipart_suggestion(
                    fluent::suggestion_escape,
                    spans
                        .into_iter()
                        .map(|(c, span)| {
                            let c = format!("{:?}", c);
                            (span, c[1..c.len() - 1].to_string())
                        })
                        .collect(),
                    Applicability::MachineApplicable,
                );
            }
            HiddenUnicodeCodepointsDiagSub::NoEscape { spans } => {
                // FIXME: in other suggestions we've reversed the inner spans of doc comments. We
                // should do the same here to provide the same good suggestions as we do for
                // literals above.
                diag.set_arg(
                    "escaped",
                    spans
                        .into_iter()
                        .map(|(c, _)| format!("{:?}", c))
                        .collect::<Vec<String>>()
                        .join(", "),
                );
                diag.note(fluent::suggestion_remove);
                diag.note(fluent::no_suggestion_note_escape);
            }
        }
    }
}

// internal.rs
#[derive(LintDiagnostic)]
#[diag(lint_default_hash_types)]
#[note]
pub struct DefaultHashTypesDiag<'a> {
    pub preferred: &'a str,
    pub used: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_query_instability)]
#[note]
pub struct QueryInstability {
    pub query: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_tykind_kind)]
pub struct TykindKind {
    #[suggestion(code = "ty", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_tykind)]
#[help]
pub struct TykindDiag;

#[derive(LintDiagnostic)]
#[diag(lint_ty_qualified)]
pub struct TyQualified {
    pub ty: String,
    #[suggestion(code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_lintpass_by_hand)]
#[help]
pub struct LintPassByHand;

#[derive(LintDiagnostic)]
#[diag(lint_non_existant_doc_keyword)]
#[help]
pub struct NonExistantDocKeyword {
    pub keyword: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_diag_out_of_impl)]
pub struct DiagOutOfImpl;

#[derive(LintDiagnostic)]
#[diag(lint_untranslatable_diag)]
pub struct UntranslatableDiag;

#[derive(LintDiagnostic)]
#[diag(lint_bad_opt_access)]
pub struct BadOptAccessDiag<'a> {
    pub msg: &'a str,
}

// let_underscore.rs
#[derive(LintDiagnostic)]
pub enum NonBindingLet {
    #[diag(lint_non_binding_let_on_sync_lock)]
    SyncLock {
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
    #[diag(lint_non_binding_let_on_drop_type)]
    DropType {
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
}

pub struct NonBindingLetSub {
    pub suggestion: Span,
    pub multi_suggestion_start: Span,
    pub multi_suggestion_end: Span,
}

impl AddToDiagnostic for NonBindingLetSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.span_suggestion_verbose(
            self.suggestion,
            fluent::lint_non_binding_let_suggestion,
            "_unused",
            Applicability::MachineApplicable,
        );
        diag.multipart_suggestion(
            fluent::lint_non_binding_let_multi_suggestion,
            vec![
                (self.multi_suggestion_start, "drop(".to_string()),
                (self.multi_suggestion_end, ")".to_string()),
            ],
            Applicability::MachineApplicable,
        );
    }
}

// levels.rs
#[derive(LintDiagnostic)]
#[diag(lint_overruled_attribute)]
pub struct OverruledAtributeLint<'a> {
    #[label]
    pub overruled: Span,
    pub lint_level: &'a str,
    pub lint_source: Symbol,
    #[subdiagnostic]
    pub sub: OverruledAttributeSub,
}

#[derive(LintDiagnostic)]
#[diag(lint_deprecated_lint_name)]
pub struct DeprecatedLintName<'a> {
    pub name: String,
    #[suggestion(code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: &'a str,
}

// FIXME: Non-translatable msg
#[derive(LintDiagnostic)]
#[diag(lint_renamed_or_removed_lint)]
pub struct RenamedOrRemovedLint<'a> {
    pub msg: &'a str,
    #[subdiagnostic]
    pub suggestion: Option<RenamedOrRemovedLintSuggestion<'a>>,
}

#[derive(Subdiagnostic)]
#[suggestion(suggestion, code = "{replace}", applicability = "machine-applicable")]
pub struct RenamedOrRemovedLintSuggestion<'a> {
    #[primary_span]
    pub suggestion: Span,
    pub replace: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_unknown_lint)]
pub struct UnknownLint {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnknownLintSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion(suggestion, code = "{replace}", applicability = "maybe-incorrect")]
pub struct UnknownLintSuggestion {
    #[primary_span]
    pub suggestion: Span,
    pub replace: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_ignored_unless_crate_specified)]
pub struct IgnoredUnlessCrateSpecified<'a> {
    pub level: &'a str,
    pub name: Symbol,
}

// methods.rs
#[derive(LintDiagnostic)]
#[diag(lint_cstring_ptr)]
#[note]
#[help]
pub struct CStringPtr {
    #[label(as_ptr_label)]
    pub as_ptr: Span,
    #[label(unwrap_label)]
    pub unwrap: Span,
}

// multiple_supertrait_upcastable.rs
#[derive(LintDiagnostic)]
#[diag(lint_multple_supertrait_upcastable)]
pub struct MultipleSupertraitUpcastable {
    pub ident: Ident,
}

// non_ascii_idents.rs
#[derive(LintDiagnostic)]
#[diag(lint_identifier_non_ascii_char)]
pub struct IdentifierNonAsciiChar;

#[derive(LintDiagnostic)]
#[diag(lint_identifier_uncommon_codepoints)]
pub struct IdentifierUncommonCodepoints;

#[derive(LintDiagnostic)]
#[diag(lint_confusable_identifier_pair)]
pub struct ConfusableIdentifierPair {
    pub existing_sym: Symbol,
    pub sym: Symbol,
    #[label]
    pub label: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_mixed_script_confusables)]
#[note(includes_note)]
#[note]
pub struct MixedScriptConfusables {
    pub set: String,
    pub includes: String,
}

// non_fmt_panic.rs
pub struct NonFmtPanicUnused {
    pub count: usize,
    pub suggestion: Option<Span>,
}

// Used because of two suggestions based on one Option<Span>
impl<'a> DecorateLint<'a, ()> for NonFmtPanicUnused {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("count", self.count);
        diag.note(fluent::note);
        if let Some(span) = self.suggestion {
            diag.span_suggestion(
                span.shrink_to_hi(),
                fluent::add_args_suggestion,
                ", ...",
                Applicability::HasPlaceholders,
            );
            diag.span_suggestion(
                span.shrink_to_lo(),
                fluent::add_fmt_suggestion,
                "\"{}\", ",
                Applicability::MachineApplicable,
            );
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_non_fmt_panic_unused
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_non_fmt_panic_braces)]
#[note]
pub struct NonFmtPanicBraces {
    pub count: usize,
    #[suggestion(code = "\"{{}}\", ", applicability = "machine-applicable")]
    pub suggestion: Option<Span>,
}

// nonstandard_style.rs
#[derive(LintDiagnostic)]
#[diag(lint_non_camel_case_type)]
pub struct NonCamelCaseType<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonCamelCaseTypeSub,
}

#[derive(Subdiagnostic)]
pub enum NonCamelCaseTypeSub {
    #[label(label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(suggestion, code = "{replace}", applicability = "maybe-incorrect")]
    Suggestion {
        #[primary_span]
        span: Span,
        replace: String,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_non_snake_case)]
pub struct NonSnakeCaseDiag<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    pub sc: String,
    #[subdiagnostic]
    pub sub: NonSnakeCaseDiagSub,
}

pub enum NonSnakeCaseDiagSub {
    Label { span: Span },
    Help,
    RenameOrConvertSuggestion { span: Span, suggestion: Ident },
    ConvertSuggestion { span: Span, suggestion: String },
    SuggestionAndNote { span: Span },
}

impl AddToDiagnostic for NonSnakeCaseDiagSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        match self {
            NonSnakeCaseDiagSub::Label { span } => {
                diag.span_label(span, fluent::label);
            }
            NonSnakeCaseDiagSub::Help => {
                diag.help(fluent::help);
            }
            NonSnakeCaseDiagSub::ConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::RenameOrConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::rename_or_convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::SuggestionAndNote { span } => {
                diag.note(fluent::cannot_convert_note);
                diag.span_suggestion(
                    span,
                    fluent::rename_suggestion,
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_non_upper_case_global)]
pub struct NonUpperCaseGlobal<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonUpperCaseGlobalSub,
}

#[derive(Subdiagnostic)]
pub enum NonUpperCaseGlobalSub {
    #[label(label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(suggestion, code = "{replace}", applicability = "maybe-incorrect")]
    Suggestion {
        #[primary_span]
        span: Span,
        replace: String,
    },
}

// noop_method_call.rs
#[derive(LintDiagnostic)]
#[diag(lint_noop_method_call)]
#[note]
pub struct NoopMethodCallDiag<'a> {
    pub method: Symbol,
    pub receiver_ty: Ty<'a>,
    #[label]
    pub label: Span,
}

// pass_by_value.rs
#[derive(LintDiagnostic)]
#[diag(lint_pass_by_value)]
pub struct PassByValueDiag {
    pub ty: String,
    #[suggestion(code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

// redundant_semicolon.rs
#[derive(LintDiagnostic)]
#[diag(lint_redundant_semicolons)]
pub struct RedundantSemicolonsDiag {
    pub multiple: bool,
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

// traits.rs
pub struct DropTraitConstraintsDiag<'a> {
    pub predicate: Predicate<'a>,
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> DecorateLint<'a, ()> for DropTraitConstraintsDiag<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("predicate", self.predicate);
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id))
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_drop_trait_constraints
    }
}

pub struct DropGlue<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> DecorateLint<'a, ()> for DropGlue<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id))
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_drop_glue
    }
}

// types.rs
#[derive(LintDiagnostic)]
#[diag(lint_range_endpoint_out_of_range)]
pub struct RangeEndpointOutOfRange<'a> {
    pub ty: &'a str,
    #[suggestion(code = "{start}..={literal}{suffix}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub start: String,
    pub literal: u128,
    pub suffix: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_bin_hex)]
pub struct OverflowingBinHex<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub dec: u128,
    pub actually: String,
    #[subdiagnostic]
    pub sign: OverflowingBinHexSign,
    #[subdiagnostic]
    pub sub: Option<OverflowingBinHexSub<'a>>,
}

pub enum OverflowingBinHexSign {
    Positive,
    Negative,
}

impl AddToDiagnostic for OverflowingBinHexSign {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        match self {
            OverflowingBinHexSign::Positive => {
                diag.note(fluent::positive_note);
            }
            OverflowingBinHexSign::Negative => {
                diag.note(fluent::negative_note);
                diag.note(fluent::negative_becomes_note);
            }
        }
    }
}

#[derive(Subdiagnostic)]
pub enum OverflowingBinHexSub<'a> {
    #[suggestion(
        suggestion,
        code = "{sans_suffix}{suggestion_ty}",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion_ty: &'a str,
        sans_suffix: &'a str,
    },
    #[help(help)]
    Help { suggestion_ty: &'a str },
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_int)]
#[note]
pub struct OverflowingInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: i128,
    pub max: u128,
    #[subdiagnostic]
    pub help: Option<OverflowingIntHelp<'a>>,
}

#[derive(Subdiagnostic)]
#[help(help)]
pub struct OverflowingIntHelp<'a> {
    pub suggestion_ty: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_only_cast_u8_to_char)]
pub struct OnlyCastu8ToChar {
    #[suggestion(code = "'\\u{{{literal:X}}}'", applicability = "machine-applicable")]
    pub span: Span,
    pub literal: u128,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_uint)]
#[note]
pub struct OverflowingUInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: u128,
    pub max: u128,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_literal)]
#[note]
pub struct OverflowingLiteral<'a> {
    pub ty: &'a str,
    pub lit: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_comparisons)]
pub struct UnusedComparisons;

pub struct ImproperCTypes<'a> {
    pub ty: Ty<'a>,
    pub desc: &'a str,
    pub label: Span,
    pub help: Option<DiagnosticMessage>,
    pub note: DiagnosticMessage,
    pub span_note: Option<Span>,
}

// Used because of the complexity of Option<DiagnosticMessage>, DiagnosticMessage, and Option<Span>
impl<'a> DecorateLint<'a, ()> for ImproperCTypes<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("ty", self.ty);
        diag.set_arg("desc", self.desc);
        diag.span_label(self.label, fluent::label);
        if let Some(help) = self.help {
            diag.help(help);
        }
        diag.note(self.note);
        if let Some(note) = self.span_note {
            diag.span_note(note, fluent::note);
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_improper_ctypes
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_variant_size_differences)]
pub struct VariantSizeDifferencesDiag {
    pub largest: u64,
}

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_load)]
#[help]
pub struct AtomicOrderingLoad;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_store)]
#[help]
pub struct AtomicOrderingStore;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_fence)]
#[help]
pub struct AtomicOrderingFence;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_invalid)]
#[help]
pub struct InvalidAtomicOrderingDiag {
    pub method: Symbol,
    #[label]
    pub fail_order_arg_span: Span,
}

// unused.rs
#[derive(LintDiagnostic)]
#[diag(lint_unused_op)]
pub struct UnusedOp<'a> {
    pub op: &'a str,
    #[label]
    pub label: Span,
    #[suggestion(style = "verbose", code = "let _ = ", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_result)]
pub struct UnusedResult<'a> {
    pub ty: Ty<'a>,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_closure)]
#[note]
pub struct UnusedClosure<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_generator)]
#[note]
pub struct UnusedGenerator<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the pre/post
// strings
pub struct UnusedDef<'a, 'b> {
    pub pre: &'a str,
    pub post: &'a str,
    pub cx: &'a LateContext<'b>,
    pub def_id: DefId,
    pub note: Option<Symbol>,
    pub suggestion: Option<UnusedDefSuggestion>,
}

#[derive(Subdiagnostic)]
pub enum UnusedDefSuggestion {
    #[suggestion(
        suggestion,
        style = "verbose",
        code = "let _ = ",
        applicability = "machine-applicable"
    )]
    Default {
        #[primary_span]
        span: Span,
    },
}

// Needed because of def_path_str
impl<'a> DecorateLint<'a, ()> for UnusedDef<'_, '_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("pre", self.pre);
        diag.set_arg("post", self.post);
        diag.set_arg("def", self.cx.tcx.def_path_str(self.def_id));
        // check for #[must_use = "..."]
        if let Some(note) = self.note {
            diag.note(note.as_str());
        }
        if let Some(sugg) = self.suggestion {
            diag.subdiagnostic(sugg);
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_unused_def
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_path_statement_drop)]
pub struct PathStatementDrop {
    #[subdiagnostic]
    pub sub: PathStatementDropSub,
}

#[derive(Subdiagnostic)]
pub enum PathStatementDropSub {
    #[suggestion(suggestion, code = "drop({snippet});", applicability = "machine-applicable")]
    Suggestion {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help(help)]
    Help {
        #[primary_span]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_path_statement_no_effect)]
pub struct PathStatementNoEffect;

#[derive(LintDiagnostic)]
#[diag(lint_unused_delim)]
pub struct UnusedDelim<'a> {
    pub delim: &'static str,
    pub item: &'a str,
    #[subdiagnostic]
    pub suggestion: Option<UnusedDelimSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion, applicability = "machine-applicable")]
pub struct UnusedDelimSuggestion {
    #[suggestion_part(code = "{start_replace}")]
    pub start_span: Span,
    pub start_replace: &'static str,
    #[suggestion_part(code = "{end_replace}")]
    pub end_span: Span,
    pub end_replace: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_import_braces)]
pub struct UnusedImportBracesDiag {
    pub node: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_allocation)]
pub struct UnusedAllocationDiag;

#[derive(LintDiagnostic)]
#[diag(lint_unused_allocation_mut)]
pub struct UnusedAllocationMutDiag;
