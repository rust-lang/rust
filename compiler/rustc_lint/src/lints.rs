#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
use std::num::NonZero;

use crate::errors::RequestedLevel;
use crate::fluent_generated as fluent;
use rustc_errors::{
    codes::*, Applicability, Diag, DiagArgValue, DiagMessage, DiagStyledString,
    ElidedLifetimeInPathSubdiag, EmissionGuarantee, LintDiagnostic, MultiSpan, SubdiagMessageOp,
    Subdiagnostic, SuggestionStyle,
};
use rustc_hir::{def::Namespace, def_id::DefId};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{
    inhabitedness::InhabitedPredicate, Clause, PolyExistentialTraitRef, Ty, TyCtxt,
};
use rustc_session::{lint::AmbiguityErrorDiag, Session};
use rustc_span::{
    edition::Edition,
    sym,
    symbol::{Ident, MacroRulesNormalizedIdent},
    Span, Symbol,
};

use crate::{
    builtin::InitError, builtin::TypeAliasBounds, errors::OverruledAttributeSub, LateContext,
};

// array_into_iter.rs
#[derive(LintDiagnostic)]
#[diag(lint_shadowed_into_iter)]
pub struct ShadowedIntoIterDiag {
    pub target: &'static str,
    pub edition: &'static str,
    #[suggestion(lint_use_iter_suggestion, code = "iter", applicability = "machine-applicable")]
    pub suggestion: Span,
    #[subdiagnostic]
    pub sub: Option<ShadowedIntoIterDiagSub>,
}

#[derive(Subdiagnostic)]
pub enum ShadowedIntoIterDiagSub {
    #[suggestion(lint_remove_into_iter_suggestion, code = "", applicability = "maybe-incorrect")]
    RemoveIntoIter {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        lint_use_explicit_into_iter_suggestion,
        applicability = "maybe-incorrect"
    )]
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
    #[diag(lint_builtin_unsafe_extern_block)]
    UnsafeExternBlock,
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
    #[diag(lint_builtin_global_asm)]
    #[note(lint_builtin_global_macro_unsafety)]
    GlobalAsm,
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
impl<'a> LintDiagnostic<'a, ()> for BuiltinMissingDebugImpl<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.primary_message(fluent::lint_builtin_missing_debug_impl);
        diag.arg("debug", self.tcx.def_path_str(self.def_id));
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
    #[suggestion(lint_msg_suggestion, code = "", applicability = "machine-applicable")]
    Msg {
        #[primary_span]
        suggestion: Span,
        msg: &'a str,
    },
    #[suggestion(lint_default_suggestion, code = "", applicability = "machine-applicable")]
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
    #[help(lint_plain_help)]
    PlainHelp,
    #[help(lint_block_help)]
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
    pub session: &'a Session,
}

impl<'a> LintDiagnostic<'a, ()> for BuiltinUngatedAsyncFnTrackCaller<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_ungated_async_fn_track_caller);
        diag.span_label(self.label, fluent::lint_label);
        rustc_session::parse::add_feature_diagnostics(
            diag,
            self.session,
            sym::async_fn_track_caller,
        );
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

impl<'a, 'b> Subdiagnostic for SuggestChangingAssocTypes<'a, 'b> {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        // Access to associates types should use `<T as Bound>::Assoc`, which does not need a
        // bound. Let's see if this type does that.

        // We use a HIR visitor to walk the type.
        use rustc_hir::intravisit::{self, Visitor};
        struct WalkAssocTypes<'a, 'b, G: EmissionGuarantee> {
            err: &'a mut Diag<'b, G>,
        }
        impl<'a, 'b, G: EmissionGuarantee> Visitor<'_> for WalkAssocTypes<'a, 'b, G> {
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

#[derive(LintDiagnostic)]
#[diag(lint_macro_expr_fragment_specifier_2024_migration)]
pub struct MacroExprFragment2024 {
    #[suggestion(code = "expr_2021", applicability = "machine-applicable")]
    pub suggestion: Span,
}

pub struct BuiltinTypeAliasGenericBoundsSuggestion {
    pub suggestions: Vec<(Span, String)>,
}

impl Subdiagnostic for BuiltinTypeAliasGenericBoundsSuggestion {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        diag.multipart_suggestion(
            fluent::lint_suggestion,
            self.suggestions,
            Applicability::MachineApplicable,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_trivial_bounds)]
pub struct BuiltinTrivialBounds<'a> {
    pub predicate_kind_name: &'a str,
    pub predicate: Clause<'a>,
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
#[multipart_suggestion(lint_suggestion)]
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
    pub note: Option<BuiltinFeatureIssueNote>,
    #[subdiagnostic]
    pub help: Option<BuiltinIncompleteFeaturesHelp>,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_internal_features)]
#[note]
pub struct BuiltinInternalFeatures {
    pub name: Symbol,
}

#[derive(Subdiagnostic)]
#[help(lint_help)]
pub struct BuiltinIncompleteFeaturesHelp;

#[derive(Subdiagnostic)]
#[note(lint_note)]
pub struct BuiltinFeatureIssueNote {
    pub n: NonZero<u32>,
}

pub struct BuiltinUnpermittedTypeInit<'a> {
    pub msg: DiagMessage,
    pub ty: Ty<'a>,
    pub label: Span,
    pub sub: BuiltinUnpermittedTypeInitSub,
    pub tcx: TyCtxt<'a>,
}

impl<'a> LintDiagnostic<'a, ()> for BuiltinUnpermittedTypeInit<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(self.msg);
        diag.arg("ty", self.ty);
        diag.span_label(self.label, fluent::lint_builtin_unpermitted_type_init_label);
        if let InhabitedPredicate::True = self.ty.inhabited_predicate(self.tcx) {
            // Only suggest late `MaybeUninit::assume_init` initialization if the type is inhabited.
            diag.span_label(
                self.label,
                fluent::lint_builtin_unpermitted_type_init_label_suggestion,
            );
        }
        self.sub.add_to_diag(diag);
    }
}

// FIXME(davidtwco): make translatable
pub struct BuiltinUnpermittedTypeInitSub {
    pub err: InitError,
}

impl Subdiagnostic for BuiltinUnpermittedTypeInitSub {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
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
        #[label(lint_previous_decl_label)]
        previous_decl_label: Span,
        #[label(lint_mismatch_label)]
        mismatch_label: Span,
        #[subdiagnostic]
        sub: BuiltinClashingExternSub<'a>,
    },
    #[diag(lint_builtin_clashing_extern_diff_name)]
    DiffName {
        this: Symbol,
        orig: Symbol,
        #[label(lint_previous_decl_label)]
        previous_decl_label: Span,
        #[label(lint_mismatch_label)]
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

impl Subdiagnostic for BuiltinClashingExternSub<'_> {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        let mut expected_str = DiagStyledString::new();
        expected_str.push(self.expected.fn_sig(self.tcx).to_string(), false);
        let mut found_str = DiagStyledString::new();
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

// deref_into_dyn_supertrait.rs
#[derive(LintDiagnostic)]
#[diag(lint_supertrait_as_deref_target)]
pub struct SupertraitAsDerefTarget<'a> {
    pub self_ty: Ty<'a>,
    pub supertrait_principal: PolyExistentialTraitRef<'a>,
    pub target_principal: PolyExistentialTraitRef<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub label2: Option<SupertraitAsDerefTargetLabel>,
}

#[derive(Subdiagnostic)]
#[label(lint_label2)]
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
#[note(lint_rationale)]
pub struct ExpectationNote {
    pub rationale: Symbol,
}

// ptr_nulls.rs
#[derive(LintDiagnostic)]
pub enum PtrNullChecksDiag<'a> {
    #[diag(lint_ptr_null_checks_fn_ptr)]
    #[help(lint_help)]
    FnPtr {
        orig_ty: Ty<'a>,
        #[label]
        label: Span,
    },
    #[diag(lint_ptr_null_checks_ref)]
    Ref {
        orig_ty: Ty<'a>,
        #[label]
        label: Span,
    },
    #[diag(lint_ptr_null_checks_fn_ret)]
    FnRet { fn_name: Ident },
}

// for_loops_over_fallibles.rs
#[derive(LintDiagnostic)]
#[diag(lint_for_loops_over_fallibles)]
pub struct ForLoopsOverFalliblesDiag<'a> {
    pub article: &'static str,
    pub ref_prefix: &'static str,
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
    #[suggestion(lint_remove_next, code = ".by_ref()", applicability = "maybe-incorrect")]
    RemoveNext {
        #[primary_span]
        suggestion: Span,
        recv_snip: String,
    },
    #[multipart_suggestion(lint_use_while_let, applicability = "maybe-incorrect")]
    UseWhileLet {
        #[suggestion_part(code = "while let {var}(")]
        start_span: Span,
        #[suggestion_part(code = ") = ")]
        end_span: Span,
        var: &'a str,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(lint_use_question_mark, code = "?", applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesQuestionMark {
    #[primary_span]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesSuggestion<'a> {
    pub var: &'a str,
    #[suggestion_part(code = "if let {var}(")]
    pub start_span: Span,
    #[suggestion_part(code = ") = ")]
    pub end_span: Span,
}

#[derive(Subdiagnostic)]
pub enum UseLetUnderscoreIgnoreSuggestion {
    #[note(lint_use_let_underscore_ignore_suggestion)]
    Note,
    #[multipart_suggestion(
        lint_use_let_underscore_ignore_suggestion,
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Suggestion {
        #[suggestion_part(code = "let _ = ")]
        start_span: Span,
        #[suggestion_part(code = "")]
        end_span: Span,
    },
}

// drop_forget_useless.rs
#[derive(LintDiagnostic)]
#[diag(lint_dropping_references)]
pub struct DropRefDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag(lint_dropping_copy_types)]
pub struct DropCopyDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag(lint_forgetting_references)]
pub struct ForgetRefDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag(lint_forgetting_copy_types)]
pub struct ForgetCopyDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag(lint_undropped_manually_drops)]
pub struct UndroppedManuallyDropsDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub suggestion: UndroppedManuallyDropsSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
pub struct UndroppedManuallyDropsSuggestion {
    #[suggestion_part(code = "std::mem::ManuallyDrop::into_inner(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

// invalid_from_utf8.rs
#[derive(LintDiagnostic)]
pub enum InvalidFromUtf8Diag {
    #[diag(lint_invalid_from_utf8_unchecked)]
    Unchecked {
        method: String,
        valid_up_to: usize,
        #[label]
        label: Span,
    },
    #[diag(lint_invalid_from_utf8_checked)]
    Checked {
        method: String,
        valid_up_to: usize,
        #[label]
        label: Span,
    },
}

// reference_casting.rs
#[derive(LintDiagnostic)]
pub enum InvalidReferenceCastingDiag<'tcx> {
    #[diag(lint_invalid_reference_casting_borrow_as_mut)]
    #[note(lint_invalid_reference_casting_note_book)]
    BorrowAsMut {
        #[label]
        orig_cast: Option<Span>,
        #[note(lint_invalid_reference_casting_note_ty_has_interior_mutability)]
        ty_has_interior_mutability: Option<()>,
    },
    #[diag(lint_invalid_reference_casting_assign_to_ref)]
    #[note(lint_invalid_reference_casting_note_book)]
    AssignToRef {
        #[label]
        orig_cast: Option<Span>,
        #[note(lint_invalid_reference_casting_note_ty_has_interior_mutability)]
        ty_has_interior_mutability: Option<()>,
    },
    #[diag(lint_invalid_reference_casting_bigger_layout)]
    #[note(lint_layout)]
    BiggerLayout {
        #[label]
        orig_cast: Option<Span>,
        #[label(lint_alloc)]
        alloc: Span,
        from_ty: Ty<'tcx>,
        from_size: u64,
        to_ty: Ty<'tcx>,
        to_size: u64,
    },
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

impl Subdiagnostic for HiddenUnicodeCodepointsDiagLabels {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        for (c, span) in self.spans {
            diag.span_label(span, format!("{c:?}"));
        }
    }
}

pub enum HiddenUnicodeCodepointsDiagSub {
    Escape { spans: Vec<(char, Span)> },
    NoEscape { spans: Vec<(char, Span)> },
}

// Used because of multiple multipart_suggestion and note
impl Subdiagnostic for HiddenUnicodeCodepointsDiagSub {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        match self {
            HiddenUnicodeCodepointsDiagSub::Escape { spans } => {
                diag.multipart_suggestion_with_style(
                    fluent::lint_suggestion_remove,
                    spans.iter().map(|(_, span)| (*span, "".to_string())).collect(),
                    Applicability::MachineApplicable,
                    SuggestionStyle::HideCodeAlways,
                );
                diag.multipart_suggestion(
                    fluent::lint_suggestion_escape,
                    spans
                        .into_iter()
                        .map(|(c, span)| {
                            let c = format!("{c:?}");
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
                diag.arg(
                    "escaped",
                    spans
                        .into_iter()
                        .map(|(c, _)| format!("{c:?}"))
                        .collect::<Vec<String>>()
                        .join(", "),
                );
                diag.note(fluent::lint_suggestion_remove);
                diag.note(fluent::lint_no_suggestion_note_escape);
            }
        }
    }
}

// map_unit_fn.rs
#[derive(LintDiagnostic)]
#[diag(lint_map_unit_fn)]
#[note]
pub struct MappingToUnit {
    #[label(lint_function_label)]
    pub function_label: Span,
    #[label(lint_argument_label)]
    pub argument_label: Span,
    #[label(lint_map_label)]
    pub map_label: Span,
    #[suggestion(style = "verbose", code = "{replace}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
    pub replace: String,
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
#[diag(lint_span_use_eq_ctxt)]
pub struct SpanUseEqCtxtDiag;

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
#[diag(lint_non_glob_import_type_ir_inherent)]
pub struct NonGlobImportTypeIrInherent {
    #[suggestion(code = "{snippet}", applicability = "maybe-incorrect")]
    pub suggestion: Option<Span>,
    pub snippet: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(lint_lintpass_by_hand)]
#[help]
pub struct LintPassByHand;

#[derive(LintDiagnostic)]
#[diag(lint_non_existent_doc_keyword)]
#[help]
pub struct NonExistentDocKeyword {
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
    pub drop_fn_start_end: Option<(Span, Span)>,
    pub is_assign_desugar: bool,
}

impl Subdiagnostic for NonBindingLetSub {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        let can_suggest_binding = self.drop_fn_start_end.is_some() || !self.is_assign_desugar;

        if can_suggest_binding {
            let prefix = if self.is_assign_desugar { "let " } else { "" };
            diag.span_suggestion_verbose(
                self.suggestion,
                fluent::lint_non_binding_let_suggestion,
                format!("{prefix}_unused"),
                Applicability::MachineApplicable,
            );
        } else {
            diag.span_help(self.suggestion, fluent::lint_non_binding_let_suggestion);
        }
        if let Some(drop_fn_start_end) = self.drop_fn_start_end {
            diag.multipart_suggestion(
                fluent::lint_non_binding_let_multi_suggestion,
                vec![
                    (drop_fn_start_end.0, "drop(".to_string()),
                    (drop_fn_start_end.1, ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
        } else {
            diag.help(fluent::lint_non_binding_let_multi_drop_fn);
        }
    }
}

// levels.rs
#[derive(LintDiagnostic)]
#[diag(lint_overruled_attribute)]
pub struct OverruledAttributeLint<'a> {
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

#[derive(LintDiagnostic)]
#[diag(lint_deprecated_lint_name)]
#[help]
pub struct DeprecatedLintNameFromCommandLine<'a> {
    pub name: String,
    pub replace: &'a str,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_renamed_lint)]
pub struct RenamedLint<'a> {
    pub name: &'a str,
    #[subdiagnostic]
    pub suggestion: RenamedLintSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub enum RenamedLintSuggestion<'a> {
    #[suggestion(lint_suggestion, code = "{replace}", applicability = "machine-applicable")]
    WithSpan {
        #[primary_span]
        suggestion: Span,
        replace: &'a str,
    },
    #[help(lint_help)]
    WithoutSpan { replace: &'a str },
}

#[derive(LintDiagnostic)]
#[diag(lint_renamed_lint)]
pub struct RenamedLintFromCommandLine<'a> {
    pub name: &'a str,
    #[subdiagnostic]
    pub suggestion: RenamedLintSuggestion<'a>,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_removed_lint)]
pub struct RemovedLint<'a> {
    pub name: &'a str,
    pub reason: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_removed_lint)]
pub struct RemovedLintFromCommandLine<'a> {
    pub name: &'a str,
    pub reason: &'a str,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_unknown_lint)]
pub struct UnknownLint {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnknownLintSuggestion>,
}

#[derive(Subdiagnostic)]
pub enum UnknownLintSuggestion {
    #[suggestion(lint_suggestion, code = "{replace}", applicability = "maybe-incorrect")]
    WithSpan {
        #[primary_span]
        suggestion: Span,
        replace: Symbol,
        from_rustc: bool,
    },
    #[help(lint_help)]
    WithoutSpan { replace: Symbol, from_rustc: bool },
}

#[derive(LintDiagnostic)]
#[diag(lint_unknown_lint, code = E0602)]
pub struct UnknownLintFromCommandLine<'a> {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnknownLintSuggestion>,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
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
    #[label(lint_as_ptr_label)]
    pub as_ptr: Span,
    #[label(lint_unwrap_label)]
    pub unwrap: Span,
}

// multiple_supertrait_upcastable.rs
#[derive(LintDiagnostic)]
#[diag(lint_multiple_supertrait_upcastable)]
pub struct MultipleSupertraitUpcastable {
    pub ident: Ident,
}

// non_ascii_idents.rs
#[derive(LintDiagnostic)]
#[diag(lint_identifier_non_ascii_char)]
pub struct IdentifierNonAsciiChar;

#[derive(LintDiagnostic)]
#[diag(lint_identifier_uncommon_codepoints)]
#[note]
pub struct IdentifierUncommonCodepoints {
    pub codepoints: Vec<char>,
    pub codepoints_len: usize,
    pub identifier_type: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(lint_confusable_identifier_pair)]
pub struct ConfusableIdentifierPair {
    pub existing_sym: Symbol,
    pub sym: Symbol,
    #[label(lint_other_use)]
    pub label: Span,
    #[label(lint_current_use)]
    pub main_label: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_mixed_script_confusables)]
#[note(lint_includes_note)]
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
impl<'a> LintDiagnostic<'a, ()> for NonFmtPanicUnused {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_non_fmt_panic_unused);
        diag.arg("count", self.count);
        diag.note(fluent::lint_note);
        if let Some(span) = self.suggestion {
            diag.span_suggestion(
                span.shrink_to_hi(),
                fluent::lint_add_args_suggestion,
                ", ...",
                Applicability::HasPlaceholders,
            );
            diag.span_suggestion(
                span.shrink_to_lo(),
                fluent::lint_add_fmt_suggestion,
                "\"{}\", ",
                Applicability::MachineApplicable,
            );
        }
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
    #[label(lint_label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(lint_suggestion, code = "{replace}", applicability = "maybe-incorrect")]
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

impl Subdiagnostic for NonSnakeCaseDiagSub {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        match self {
            NonSnakeCaseDiagSub::Label { span } => {
                diag.span_label(span, fluent::lint_label);
            }
            NonSnakeCaseDiagSub::Help => {
                diag.help(fluent::lint_help);
            }
            NonSnakeCaseDiagSub::ConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::lint_convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::RenameOrConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::lint_rename_or_convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::SuggestionAndNote { span } => {
                diag.note(fluent::lint_cannot_convert_note);
                diag.span_suggestion(
                    span,
                    fluent::lint_rename_suggestion,
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
    #[label(lint_label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(lint_suggestion, code = "{replace}", applicability = "maybe-incorrect")]
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
    pub orig_ty: Ty<'a>,
    pub trait_: Symbol,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub label: Span,
    #[suggestion(
        lint_derive_suggestion,
        code = "#[derive(Clone)]\n",
        applicability = "maybe-incorrect"
    )]
    pub suggest_derive: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(lint_suspicious_double_ref_deref)]
pub struct SuspiciousDoubleRefDerefDiag<'a> {
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_suspicious_double_ref_clone)]
pub struct SuspiciousDoubleRefCloneDiag<'a> {
    pub ty: Ty<'a>,
}

// non_local_defs.rs
pub enum NonLocalDefinitionsDiag {
    Impl {
        depth: u32,
        body_kind_descr: &'static str,
        body_name: String,
        cargo_update: Option<NonLocalDefinitionsCargoUpdateNote>,
        const_anon: Option<Option<Span>>,
        move_to: Option<(Span, Vec<Span>)>,
        doctest: bool,
        may_remove: Option<(Span, String)>,
        has_trait: bool,
        self_ty_str: String,
        of_trait_str: Option<String>,
        macro_to_change: Option<(String, &'static str)>,
    },
    MacroRules {
        depth: u32,
        body_kind_descr: &'static str,
        body_name: String,
        doctest: bool,
        cargo_update: Option<NonLocalDefinitionsCargoUpdateNote>,
    },
}

impl<'a> LintDiagnostic<'a, ()> for NonLocalDefinitionsDiag {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        match self {
            NonLocalDefinitionsDiag::Impl {
                depth,
                body_kind_descr,
                body_name,
                cargo_update,
                const_anon,
                move_to,
                doctest,
                may_remove,
                has_trait,
                self_ty_str,
                of_trait_str,
                macro_to_change,
            } => {
                diag.primary_message(fluent::lint_non_local_definitions_impl);
                diag.arg("depth", depth);
                diag.arg("body_kind_descr", body_kind_descr);
                diag.arg("body_name", body_name);
                diag.arg("self_ty_str", self_ty_str);
                if let Some(of_trait_str) = of_trait_str {
                    diag.arg("of_trait_str", of_trait_str);
                }

                if let Some((macro_to_change, macro_kind)) = macro_to_change {
                    diag.arg("macro_to_change", macro_to_change);
                    diag.arg("macro_kind", macro_kind);
                    diag.note(fluent::lint_macro_to_change);
                }
                if let Some(cargo_update) = cargo_update {
                    diag.subdiagnostic(cargo_update);
                }

                if has_trait {
                    diag.note(fluent::lint_bounds);
                    diag.note(fluent::lint_with_trait);
                } else {
                    diag.note(fluent::lint_without_trait);
                }

                if let Some((move_help, may_move)) = move_to {
                    let mut ms = MultiSpan::from_span(move_help);
                    for sp in may_move {
                        ms.push_span_label(sp, fluent::lint_non_local_definitions_may_move);
                    }
                    diag.span_help(ms, fluent::lint_non_local_definitions_impl_move_help);
                }
                if doctest {
                    diag.help(fluent::lint_doctest);
                }

                if let Some((span, part)) = may_remove {
                    diag.arg("may_remove_part", part);
                    diag.span_suggestion(
                        span,
                        fluent::lint_remove_help,
                        "",
                        Applicability::MaybeIncorrect,
                    );
                }

                if let Some(const_anon) = const_anon {
                    diag.note(fluent::lint_exception);
                    if let Some(const_anon) = const_anon {
                        diag.span_suggestion(
                            const_anon,
                            fluent::lint_const_anon,
                            "_",
                            Applicability::MachineApplicable,
                        );
                    }
                }

                diag.note(fluent::lint_non_local_definitions_deprecation);
            }
            NonLocalDefinitionsDiag::MacroRules {
                depth,
                body_kind_descr,
                body_name,
                doctest,
                cargo_update,
            } => {
                diag.primary_message(fluent::lint_non_local_definitions_macro_rules);
                diag.arg("depth", depth);
                diag.arg("body_kind_descr", body_kind_descr);
                diag.arg("body_name", body_name);

                if doctest {
                    diag.help(fluent::lint_help_doctest);
                } else {
                    diag.help(fluent::lint_help);
                }

                diag.note(fluent::lint_non_local);
                diag.note(fluent::lint_non_local_definitions_deprecation);

                if let Some(cargo_update) = cargo_update {
                    diag.subdiagnostic(cargo_update);
                }
            }
        }
    }
}

#[derive(Subdiagnostic)]
#[note(lint_non_local_definitions_cargo_update)]
pub struct NonLocalDefinitionsCargoUpdateNote {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
    pub crate_name: Symbol,
}

// precedence.rs
#[derive(LintDiagnostic)]
#[diag(lint_ambiguous_negative_literals)]
#[note(lint_example)]
pub struct AmbiguousNegativeLiteralsDiag {
    #[subdiagnostic]
    pub negative_literal: AmbiguousNegativeLiteralsNegativeLiteralSuggestion,
    #[subdiagnostic]
    pub current_behavior: AmbiguousNegativeLiteralsCurrentBehaviorSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_negative_literal, applicability = "maybe-incorrect")]
pub struct AmbiguousNegativeLiteralsNegativeLiteralSuggestion {
    #[suggestion_part(code = "(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_current_behavior, applicability = "maybe-incorrect")]
pub struct AmbiguousNegativeLiteralsCurrentBehaviorSuggestion {
    #[suggestion_part(code = "(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
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
    pub predicate: Clause<'a>,
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for DropTraitConstraintsDiag<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_drop_trait_constraints);
        diag.arg("predicate", self.predicate);
        diag.arg("needs_drop", self.tcx.def_path_str(self.def_id));
    }
}

pub struct DropGlue<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for DropGlue<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_drop_glue);
        diag.arg("needs_drop", self.tcx.def_path_str(self.def_id));
    }
}

// types.rs
#[derive(LintDiagnostic)]
#[diag(lint_range_endpoint_out_of_range)]
pub struct RangeEndpointOutOfRange<'a> {
    pub ty: &'a str,
    #[subdiagnostic]
    pub sub: UseInclusiveRange<'a>,
}

#[derive(Subdiagnostic)]
pub enum UseInclusiveRange<'a> {
    #[suggestion(
        lint_range_use_inclusive_range,
        code = "{start}..={literal}{suffix}",
        applicability = "machine-applicable"
    )]
    WithoutParen {
        #[primary_span]
        sugg: Span,
        start: String,
        literal: u128,
        suffix: &'a str,
    },
    #[multipart_suggestion(lint_range_use_inclusive_range, applicability = "machine-applicable")]
    WithParen {
        #[suggestion_part(code = "=")]
        eq_sugg: Span,
        #[suggestion_part(code = "{literal}{suffix}")]
        lit_sugg: Span,
        literal: u128,
        suffix: &'a str,
    },
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
    #[subdiagnostic]
    pub sign_bit_sub: Option<OverflowingBinHexSignBitSub<'a>>,
}

pub enum OverflowingBinHexSign {
    Positive,
    Negative,
}

impl Subdiagnostic for OverflowingBinHexSign {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        match self {
            OverflowingBinHexSign::Positive => {
                diag.note(fluent::lint_positive_note);
            }
            OverflowingBinHexSign::Negative => {
                diag.note(fluent::lint_negative_note);
                diag.note(fluent::lint_negative_becomes_note);
            }
        }
    }
}

#[derive(Subdiagnostic)]
pub enum OverflowingBinHexSub<'a> {
    #[suggestion(
        lint_suggestion,
        code = "{sans_suffix}{suggestion_ty}",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion_ty: &'a str,
        sans_suffix: &'a str,
    },
    #[help(lint_help)]
    Help { suggestion_ty: &'a str },
}

#[derive(Subdiagnostic)]
#[suggestion(
    lint_sign_bit_suggestion,
    code = "{lit_no_suffix}{uint_ty} as {int_ty}",
    applicability = "maybe-incorrect"
)]
pub struct OverflowingBinHexSignBitSub<'a> {
    #[primary_span]
    pub span: Span,
    pub lit_no_suffix: &'a str,
    pub negative_val: String,
    pub uint_ty: &'a str,
    pub int_ty: &'a str,
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
#[help(lint_help)]
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

#[derive(LintDiagnostic)]
pub enum InvalidNanComparisons {
    #[diag(lint_invalid_nan_comparisons_eq_ne)]
    EqNe {
        #[subdiagnostic]
        suggestion: Option<InvalidNanComparisonsSuggestion>,
    },
    #[diag(lint_invalid_nan_comparisons_lt_le_gt_ge)]
    LtLeGtGe,
}

#[derive(Subdiagnostic)]
pub enum InvalidNanComparisonsSuggestion {
    #[multipart_suggestion(
        lint_suggestion,
        style = "verbose",
        applicability = "machine-applicable"
    )]
    Spanful {
        #[suggestion_part(code = "!")]
        neg: Option<Span>,
        #[suggestion_part(code = ".is_nan()")]
        float: Span,
        #[suggestion_part(code = "")]
        nan_plus_binop: Span,
    },
    #[help(lint_suggestion)]
    Spanless,
}

#[derive(LintDiagnostic)]
pub enum AmbiguousWidePointerComparisons<'a> {
    #[diag(lint_ambiguous_wide_pointer_comparisons)]
    Spanful {
        #[subdiagnostic]
        addr_suggestion: AmbiguousWidePointerComparisonsAddrSuggestion<'a>,
        #[subdiagnostic]
        addr_metadata_suggestion: Option<AmbiguousWidePointerComparisonsAddrMetadataSuggestion<'a>>,
    },
    #[diag(lint_ambiguous_wide_pointer_comparisons)]
    #[help(lint_addr_metadata_suggestion)]
    #[help(lint_addr_suggestion)]
    Spanless,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    lint_addr_metadata_suggestion,
    style = "verbose",
    // FIXME(#53934): make machine-applicable again
    applicability = "maybe-incorrect"
)]
pub struct AmbiguousWidePointerComparisonsAddrMetadataSuggestion<'a> {
    pub ne: &'a str,
    pub deref_left: &'a str,
    pub deref_right: &'a str,
    pub l_modifiers: &'a str,
    pub r_modifiers: &'a str,
    #[suggestion_part(code = "{ne}std::ptr::eq({deref_left}")]
    pub left: Span,
    #[suggestion_part(code = "{l_modifiers}, {deref_right}")]
    pub middle: Span,
    #[suggestion_part(code = "{r_modifiers})")]
    pub right: Span,
}

#[derive(Subdiagnostic)]
pub enum AmbiguousWidePointerComparisonsAddrSuggestion<'a> {
    #[multipart_suggestion(
        lint_addr_suggestion,
        style = "verbose",
        // FIXME(#53934): make machine-applicable again
        applicability = "maybe-incorrect"
    )]
    AddrEq {
        ne: &'a str,
        deref_left: &'a str,
        deref_right: &'a str,
        l_modifiers: &'a str,
        r_modifiers: &'a str,
        #[suggestion_part(code = "{ne}std::ptr::addr_eq({deref_left}")]
        left: Span,
        #[suggestion_part(code = "{l_modifiers}, {deref_right}")]
        middle: Span,
        #[suggestion_part(code = "{r_modifiers})")]
        right: Span,
    },
    #[multipart_suggestion(
        lint_addr_suggestion,
        style = "verbose",
        // FIXME(#53934): make machine-applicable again
        applicability = "maybe-incorrect"
    )]
    Cast {
        deref_left: &'a str,
        deref_right: &'a str,
        paren_left: &'a str,
        paren_right: &'a str,
        l_modifiers: &'a str,
        r_modifiers: &'a str,
        #[suggestion_part(code = "({deref_left}")]
        left_before: Option<Span>,
        #[suggestion_part(code = "{l_modifiers}{paren_left}.cast::<()>()")]
        left_after: Span,
        #[suggestion_part(code = "({deref_right}")]
        right_before: Option<Span>,
        #[suggestion_part(code = "{r_modifiers}{paren_right}.cast::<()>()")]
        right_after: Span,
    },
}

pub struct ImproperCTypes<'a> {
    pub ty: Ty<'a>,
    pub desc: &'a str,
    pub label: Span,
    pub help: Option<DiagMessage>,
    pub note: DiagMessage,
    pub span_note: Option<Span>,
}

// Used because of the complexity of Option<DiagMessage>, DiagMessage, and Option<Span>
impl<'a> LintDiagnostic<'a, ()> for ImproperCTypes<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_improper_ctypes);
        diag.arg("ty", self.ty);
        diag.arg("desc", self.desc);
        diag.span_label(self.label, fluent::lint_label);
        if let Some(help) = self.help {
            diag.help(help);
        }
        diag.note(self.note);
        if let Some(note) = self.span_note {
            diag.span_note(note, fluent::lint_note);
        }
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
    #[subdiagnostic]
    pub suggestion: UnusedOpSuggestion,
}

#[derive(Subdiagnostic)]
pub enum UnusedOpSuggestion {
    #[suggestion(
        lint_suggestion,
        style = "verbose",
        code = "let _ = ",
        applicability = "maybe-incorrect"
    )]
    NormalExpr {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(lint_suggestion, style = "verbose", applicability = "maybe-incorrect")]
    BlockTailExpr {
        #[suggestion_part(code = "let _ = ")]
        before_span: Span,
        #[suggestion_part(code = ";")]
        after_span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_result)]
pub struct UnusedResult<'a> {
    pub ty: Ty<'a>,
}

// FIXME(davidtwco): this isn't properly translatable because of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_closure)]
#[note]
pub struct UnusedClosure<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable because of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_coroutine)]
#[note]
pub struct UnusedCoroutine<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable because of the pre/post
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
        lint_suggestion,
        style = "verbose",
        code = "let _ = ",
        applicability = "maybe-incorrect"
    )]
    NormalExpr {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(lint_suggestion, style = "verbose", applicability = "maybe-incorrect")]
    BlockTailExpr {
        #[suggestion_part(code = "let _ = ")]
        before_span: Span,
        #[suggestion_part(code = ";")]
        after_span: Span,
    },
}

// Needed because of def_path_str
impl<'a> LintDiagnostic<'a, ()> for UnusedDef<'_, '_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_unused_def);
        diag.arg("pre", self.pre);
        diag.arg("post", self.post);
        diag.arg("def", self.cx.tcx.def_path_str(self.def_id));
        // check for #[must_use = "..."]
        if let Some(note) = self.note {
            diag.note(note.to_string());
        }
        if let Some(sugg) = self.suggestion {
            diag.subdiagnostic(sugg);
        }
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
    #[suggestion(lint_suggestion, code = "drop({snippet});", applicability = "machine-applicable")]
    Suggestion {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help(lint_help)]
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
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
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

pub struct AsyncFnInTraitDiag {
    pub sugg: Option<Vec<(Span, String)>>,
}

impl<'a> LintDiagnostic<'a, ()> for AsyncFnInTraitDiag {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(fluent::lint_async_fn_in_trait);
        diag.note(fluent::lint_note);
        if let Some(sugg) = self.sugg {
            diag.multipart_suggestion(fluent::lint_suggestion, sugg, Applicability::MaybeIncorrect);
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_unit_bindings)]
pub struct UnitBindingsDiag {
    #[label]
    pub label: Span,
}

#[derive(LintDiagnostic)]
pub enum InvalidAsmLabel {
    #[diag(lint_invalid_asm_label_named)]
    #[help]
    #[note]
    Named {
        #[note(lint_invalid_asm_label_no_span)]
        missing_precise_span: bool,
    },
    #[diag(lint_invalid_asm_label_format_arg)]
    #[help]
    #[note(lint_note1)]
    #[note(lint_note2)]
    FormatArg {
        #[note(lint_invalid_asm_label_no_span)]
        missing_precise_span: bool,
    },
    #[diag(lint_invalid_asm_label_binary)]
    #[help]
    #[note(lint_note1)]
    #[note(lint_note2)]
    Binary {
        #[note(lint_invalid_asm_label_no_span)]
        missing_precise_span: bool,
        // hack to get a label on the whole span, must match the emitted span
        #[label]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub enum UnexpectedCfgCargoHelp {
    #[help(lint_unexpected_cfg_add_cargo_feature)]
    #[help(lint_unexpected_cfg_add_cargo_toml_lint_cfg)]
    LintCfg { cargo_toml_lint_cfg: String },
    #[help(lint_unexpected_cfg_add_cargo_feature)]
    #[help(lint_unexpected_cfg_add_cargo_toml_lint_cfg)]
    #[help(lint_unexpected_cfg_add_build_rs_println)]
    LintCfgAndBuildRs { cargo_toml_lint_cfg: String, build_rs_println: String },
}

impl UnexpectedCfgCargoHelp {
    fn cargo_toml_lint_cfg(unescaped: &str) -> String {
        format!(
            "\n [lints.rust]\n unexpected_cfgs = {{ level = \"warn\", check-cfg = ['{unescaped}'] }}"
        )
    }

    pub fn lint_cfg(unescaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfg {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
        }
    }

    pub fn lint_cfg_and_build_rs(unescaped: &str, escaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfgAndBuildRs {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
            build_rs_println: format!("println!(\"cargo::rustc-check-cfg={escaped}\");"),
        }
    }
}

#[derive(Subdiagnostic)]
#[help(lint_unexpected_cfg_add_cmdline_arg)]
pub struct UnexpectedCfgRustcHelp {
    pub cmdline_arg: String,
}

impl UnexpectedCfgRustcHelp {
    pub fn new(unescaped: &str) -> Self {
        Self { cmdline_arg: format!("--check-cfg={unescaped}") }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_unexpected_cfg_name)]
pub struct UnexpectedCfgName {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_name::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_name::InvocationHelp,

    pub name: Symbol,
}

pub mod unexpected_cfg_name {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Span, Symbol};

    #[derive(Subdiagnostic)]
    pub enum CodeSuggestion {
        #[help(lint_unexpected_cfg_define_features)]
        DefineFeatures,
        #[suggestion(
            lint_unexpected_cfg_name_similar_name_value,
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameAndValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            lint_unexpected_cfg_name_similar_name_no_value,
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameNoValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            lint_unexpected_cfg_name_similar_name_different_values,
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameDifferentValues {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        #[suggestion(
            lint_unexpected_cfg_name_similar_name,
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        SimilarValues {
            #[subdiagnostic]
            with_similar_values: Vec<FoundWithSimilarValue>,
            #[subdiagnostic]
            expected_names: Option<ExpectedNames>,
        },
    }

    #[derive(Subdiagnostic)]
    #[help(lint_unexpected_cfg_name_expected_values)]
    pub struct ExpectedValues {
        pub best_match: Symbol,
        pub possibilities: DiagSymbolList,
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        lint_unexpected_cfg_name_with_similar_value,
        applicability = "maybe-incorrect",
        code = "{code}"
    )]
    pub struct FoundWithSimilarValue {
        #[primary_span]
        pub span: Span,
        pub code: String,
    }

    #[derive(Subdiagnostic)]
    #[help_once(lint_unexpected_cfg_name_expected_names)]
    pub struct ExpectedNames {
        pub possibilities: DiagSymbolList,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    pub enum InvocationHelp {
        #[note(lint_unexpected_cfg_doc_cargo)]
        Cargo {
            #[subdiagnostic]
            sub: Option<super::UnexpectedCfgCargoHelp>,
        },
        #[note(lint_unexpected_cfg_doc_rustc)]
        Rustc(#[subdiagnostic] super::UnexpectedCfgRustcHelp),
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_unexpected_cfg_value)]
pub struct UnexpectedCfgValue {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_value::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_value::InvocationHelp,

    pub has_value: bool,
    pub value: String,
}

pub mod unexpected_cfg_value {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Span, Symbol};

    #[derive(Subdiagnostic)]
    pub enum CodeSuggestion {
        ChangeValue {
            #[subdiagnostic]
            expected_values: ExpectedValues,
            #[subdiagnostic]
            suggestion: Option<ChangeValueSuggestion>,
        },
        #[note(lint_unexpected_cfg_value_no_expected_value)]
        RemoveValue {
            #[subdiagnostic]
            suggestion: Option<RemoveValueSuggestion>,

            name: Symbol,
        },
        #[note(lint_unexpected_cfg_value_no_expected_values)]
        RemoveCondition {
            #[subdiagnostic]
            suggestion: RemoveConditionSuggestion,

            name: Symbol,
        },
    }

    #[derive(Subdiagnostic)]
    pub enum ChangeValueSuggestion {
        #[suggestion(
            lint_unexpected_cfg_value_similar_name,
            code = r#""{best_match}""#,
            applicability = "maybe-incorrect"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            best_match: Symbol,
        },
        #[suggestion(
            lint_unexpected_cfg_value_specify_value,
            code = r#" = "{first_possibility}""#,
            applicability = "maybe-incorrect"
        )]
        SpecifyValue {
            #[primary_span]
            span: Span,
            first_possibility: Symbol,
        },
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        lint_unexpected_cfg_value_remove_value,
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub struct RemoveValueSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        lint_unexpected_cfg_value_remove_condition,
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub struct RemoveConditionSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[note(lint_unexpected_cfg_value_expected_values)]
    pub struct ExpectedValues {
        pub name: Symbol,
        pub have_none_possibility: bool,
        pub possibilities: DiagSymbolList,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    pub enum InvocationHelp {
        #[note(lint_unexpected_cfg_doc_cargo)]
        Cargo(#[subdiagnostic] Option<CargoHelp>),
        #[note(lint_unexpected_cfg_doc_rustc)]
        Rustc(#[subdiagnostic] Option<super::UnexpectedCfgRustcHelp>),
    }

    #[derive(Subdiagnostic)]
    pub enum CargoHelp {
        #[help(lint_unexpected_cfg_value_add_feature)]
        AddFeature {
            value: Symbol,
        },
        #[help(lint_unexpected_cfg_define_features)]
        DefineFeatures,
        Other(#[subdiagnostic] super::UnexpectedCfgCargoHelp),
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_macro_use_deprecated)]
#[help]
pub struct MacroUseDeprecated;

#[derive(LintDiagnostic)]
#[diag(lint_unused_macro_use)]
pub struct UnusedMacroUse;

#[derive(LintDiagnostic)]
#[diag(lint_private_extern_crate_reexport, code = E0365)]
pub struct PrivateExternCrateReexport {
    pub ident: Ident,
    #[suggestion(code = "pub ", style = "verbose", applicability = "maybe-incorrect")]
    pub sugg: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_label)]
pub struct UnusedLabel;

#[derive(LintDiagnostic)]
#[diag(lint_macro_is_private)]
pub struct MacroIsPrivate {
    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_macro_definition)]
pub struct UnusedMacroDefinition {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_macro_rule_never_used)]
pub struct MacroRuleNeverUsed {
    pub n: usize,
    pub name: Symbol,
}

pub struct UnstableFeature {
    pub msg: DiagMessage,
}

impl<'a> LintDiagnostic<'a, ()> for UnstableFeature {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(self.msg);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_avoid_intel_syntax)]
pub struct AvoidIntelSyntax;

#[derive(LintDiagnostic)]
#[diag(lint_avoid_att_syntax)]
pub struct AvoidAttSyntax;

#[derive(LintDiagnostic)]
#[diag(lint_incomplete_include)]
pub struct IncompleteInclude;

#[derive(LintDiagnostic)]
#[diag(lint_unnameable_test_items)]
pub struct UnnameableTestItems;

#[derive(LintDiagnostic)]
#[diag(lint_duplicate_macro_attribute)]
pub struct DuplicateMacroAttribute;

#[derive(LintDiagnostic)]
#[diag(lint_cfg_attr_no_attributes)]
pub struct CfgAttrNoAttributes;

#[derive(LintDiagnostic)]
#[diag(lint_crate_type_in_cfg_attr_deprecated)]
pub struct CrateTypeInCfgAttr;

#[derive(LintDiagnostic)]
#[diag(lint_crate_name_in_cfg_attr_deprecated)]
pub struct CrateNameInCfgAttr;

#[derive(LintDiagnostic)]
#[diag(lint_missing_fragment_specifier)]
pub struct MissingFragmentSpecifier;

#[derive(LintDiagnostic)]
#[diag(lint_metavariable_still_repeating)]
pub struct MetaVariableStillRepeating {
    pub name: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag(lint_metavariable_wrong_operator)]
pub struct MetaVariableWrongOperator;

#[derive(LintDiagnostic)]
#[diag(lint_duplicate_matcher_binding)]
pub struct DuplicateMatcherBinding;

#[derive(LintDiagnostic)]
#[diag(lint_unknown_macro_variable)]
pub struct UnknownMacroVariable {
    pub name: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_crate_dependency)]
#[help]
pub struct UnusedCrateDependency {
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_wasm_c_abi)]
pub struct WasmCAbi;

#[derive(LintDiagnostic)]
#[diag(lint_ill_formed_attribute_input)]
pub struct IllFormedAttributeInput {
    pub num_suggestions: usize,
    pub suggestions: DiagArgValue,
}

#[derive(LintDiagnostic)]
pub enum InnerAttributeUnstable {
    #[diag(lint_inner_macro_attribute_unstable)]
    InnerMacroAttribute,
    #[diag(lint_custom_inner_attribute_unstable)]
    CustomInnerAttribute,
}

#[derive(LintDiagnostic)]
#[diag(lint_unknown_diagnostic_attribute)]
pub struct UnknownDiagnosticAttribute {
    #[subdiagnostic]
    pub typo: Option<UnknownDiagnosticAttributeTypoSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    lint_unknown_diagnostic_attribute_typo_sugg,
    style = "verbose",
    code = "{typo_name}",
    applicability = "machine-applicable"
)]
pub struct UnknownDiagnosticAttributeTypoSugg {
    #[primary_span]
    pub span: Span,
    pub typo_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_unicode_text_flow)]
#[note]
pub struct UnicodeTextFlow {
    #[label]
    pub comment_span: Span,
    #[subdiagnostic]
    pub characters: Vec<UnicodeCharNoteSub>,
    #[subdiagnostic]
    pub suggestions: Option<UnicodeTextFlowSuggestion>,

    pub num_codepoints: usize,
}

#[derive(Subdiagnostic)]
#[label(lint_label_comment_char)]
pub struct UnicodeCharNoteSub {
    #[primary_span]
    pub span: Span,
    pub c_debug: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable", style = "hidden")]
pub struct UnicodeTextFlowSuggestion {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
}

#[derive(LintDiagnostic)]
#[diag(lint_abs_path_with_module)]
pub struct AbsPathWithModule {
    #[subdiagnostic]
    pub sugg: AbsPathWithModuleSugg,
}

#[derive(Subdiagnostic)]
#[suggestion(lint_suggestion, code = "{replacement}")]
pub struct AbsPathWithModuleSugg {
    #[primary_span]
    pub span: Span,
    #[applicability]
    pub applicability: Applicability,
    pub replacement: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_proc_macro_derive_resolution_fallback)]
pub struct ProcMacroDeriveResolutionFallback {
    #[label]
    pub span: Span,
    pub ns: Namespace,
    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag(lint_macro_expanded_macro_exports_accessed_by_absolute_paths)]
pub struct MacroExpandedMacroExportsAccessedByAbsolutePaths {
    #[note]
    pub definition: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_hidden_lifetime_parameters)]
pub struct ElidedLifetimesInPaths {
    #[subdiagnostic]
    pub subdiag: ElidedLifetimeInPathSubdiag,
}

#[derive(LintDiagnostic)]
#[diag(lint_invalid_crate_type_value)]
pub struct UnknownCrateTypes {
    #[subdiagnostic]
    pub sugg: Option<UnknownCrateTypesSub>,
}

#[derive(Subdiagnostic)]
#[suggestion(lint_suggestion, code = r#""{candidate}""#, applicability = "maybe-incorrect")]
pub struct UnknownCrateTypesSub {
    #[primary_span]
    pub span: Span,
    pub candidate: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_imports)]
pub struct UnusedImports {
    #[subdiagnostic]
    pub sugg: UnusedImportsSugg,
    #[help]
    pub test_module_span: Option<Span>,

    pub span_snippets: DiagArgValue,
    pub num_snippets: usize,
}

#[derive(Subdiagnostic)]
pub enum UnusedImportsSugg {
    #[suggestion(
        lint_suggestion_remove_whole_use,
        applicability = "machine-applicable",
        code = "",
        style = "tool-only"
    )]
    RemoveWholeUse {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        lint_suggestion_remove_imports,
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    RemoveImports {
        #[suggestion_part(code = "")]
        remove_spans: Vec<Span>,
        num_to_remove: usize,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_redundant_import)]
pub struct RedundantImport {
    #[subdiagnostic]
    pub subs: Vec<RedundantImportSub>,

    pub ident: Ident,
}

#[derive(Subdiagnostic)]
pub enum RedundantImportSub {
    #[label(lint_label_imported_here)]
    ImportedHere(#[primary_span] Span),
    #[label(lint_label_defined_here)]
    DefinedHere(#[primary_span] Span),
    #[label(lint_label_imported_prelude)]
    ImportedPrelude(#[primary_span] Span),
    #[label(lint_label_defined_prelude)]
    DefinedPrelude(#[primary_span] Span),
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_doc_comment)]
#[help]
pub struct UnusedDocComment {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
pub enum PatternsInFnsWithoutBody {
    #[diag(lint_pattern_in_foreign)]
    Foreign {
        #[subdiagnostic]
        sub: PatternsInFnsWithoutBodySub,
    },
    #[diag(lint_pattern_in_bodiless)]
    Bodiless {
        #[subdiagnostic]
        sub: PatternsInFnsWithoutBodySub,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(lint_remove_mut_from_pattern, code = "{ident}", applicability = "machine-applicable")]
pub struct PatternsInFnsWithoutBodySub {
    #[primary_span]
    pub span: Span,

    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag(lint_extern_without_abi)]
#[help]
pub struct MissingAbi {
    #[label]
    pub span: Span,

    pub default_abi: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(lint_legacy_derive_helpers)]
pub struct LegacyDeriveHelpers {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_or_patterns_back_compat)]
pub struct OrPatternsBackCompat {
    #[suggestion(code = "{suggestion}", applicability = "machine-applicable")]
    pub span: Span,
    pub suggestion: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_reserved_prefix)]
pub struct ReservedPrefix {
    #[label]
    pub label: Span,
    #[suggestion(code = " ", applicability = "machine-applicable")]
    pub suggestion: Span,

    pub prefix: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_builtin_attribute)]
pub struct UnusedBuiltinAttribute {
    #[note]
    pub invoc_span: Span,

    pub attr_name: Symbol,
    pub macro_name: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_trailing_semi_macro)]
pub struct TrailingMacro {
    #[note(lint_note1)]
    #[note(lint_note2)]
    pub is_trailing: bool,

    pub name: Ident,
}

#[derive(LintDiagnostic)]
#[diag(lint_break_with_label_and_loop)]
pub struct BreakWithLabelAndLoop {
    #[subdiagnostic]
    pub sub: BreakWithLabelAndLoopSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
pub struct BreakWithLabelAndLoopSub {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_deprecated_where_clause_location)]
#[note]
pub struct DeprecatedWhereClauseLocation {
    #[subdiagnostic]
    pub suggestion: DeprecatedWhereClauseLocationSugg,
}

#[derive(Subdiagnostic)]
pub enum DeprecatedWhereClauseLocationSugg {
    #[multipart_suggestion(lint_suggestion_move_to_end, applicability = "machine-applicable")]
    MoveToEnd {
        #[suggestion_part(code = "")]
        left: Span,
        #[suggestion_part(code = "{sugg}")]
        right: Span,

        sugg: String,
    },
    #[suggestion(lint_suggestion_remove_where, code = "", applicability = "machine-applicable")]
    RemoveWhere {
        #[primary_span]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_missing_unsafe_on_extern)]
pub struct MissingUnsafeOnExtern {
    #[suggestion(code = "unsafe ", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_single_use_lifetime)]
pub struct SingleUseLifetime {
    #[label(lint_label_param)]
    pub param_span: Span,
    #[label(lint_label_use)]
    pub use_span: Span,
    #[subdiagnostic]
    pub suggestion: Option<SingleUseLifetimeSugg>,

    pub ident: Ident,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
pub struct SingleUseLifetimeSugg {
    #[suggestion_part(code = "")]
    pub deletion_span: Option<Span>,
    #[suggestion_part(code = "{replace_lt}")]
    pub use_span: Span,

    pub replace_lt: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_lifetime)]
pub struct UnusedLifetime {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub deletion_span: Option<Span>,

    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag(lint_named_argument_used_positionally)]
pub struct NamedArgumentUsedPositionally {
    #[label(lint_label_named_arg)]
    pub named_arg_sp: Span,
    #[label(lint_label_position_arg)]
    pub position_label_sp: Option<Span>,
    #[suggestion(style = "verbose", code = "{name}", applicability = "maybe-incorrect")]
    pub suggestion: Option<Span>,

    pub name: String,
    pub named_arg_name: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_byte_slice_in_packed_struct_with_derive)]
#[help]
pub struct ByteSliceInPackedStructWithDerive {
    // FIXME: make this translatable
    pub ty: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_extern_crate)]
pub struct UnusedExternCrate {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub removal_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_extern_crate_not_idiomatic)]
pub struct ExternCrateNotIdiomatic {
    #[suggestion(style = "verbose", code = "{code}", applicability = "machine-applicable")]
    pub span: Span,

    pub code: &'static str,
}

// FIXME: make this translatable
pub struct AmbiguousGlobImports {
    pub ambiguity: AmbiguityErrorDiag,
}

impl<'a, G: EmissionGuarantee> LintDiagnostic<'a, G> for AmbiguousGlobImports {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, G>) {
        diag.primary_message(self.ambiguity.msg.clone());
        rustc_errors::report_ambiguity_error(diag, self.ambiguity);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_ambiguous_glob_reexport)]
pub struct AmbiguousGlobReexports {
    #[label(lint_label_first_reexport)]
    pub first_reexport: Span,
    #[label(lint_label_duplicate_reexport)]
    pub duplicate_reexport: Span,

    pub name: String,
    // FIXME: make this translatable
    pub namespace: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_hidden_glob_reexport)]
pub struct HiddenGlobReexports {
    #[note(lint_note_glob_reexport)]
    pub glob_reexport: Span,
    #[note(lint_note_private_item)]
    pub private_item: Span,

    pub name: String,
    // FIXME: make this translatable
    pub namespace: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unnecessary_qualification)]
pub struct UnusedQualifications {
    #[suggestion(style = "verbose", code = "", applicability = "machine-applicable")]
    pub removal_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_associated_const_elided_lifetime)]
pub struct AssociatedConstElidedLifetime {
    #[suggestion(style = "verbose", code = "{code}", applicability = "machine-applicable")]
    pub span: Span,

    pub code: &'static str,
    pub elided: bool,
    #[note]
    pub lifetimes_in_scope: MultiSpan,
}

#[derive(LintDiagnostic)]
#[diag(lint_redundant_import_visibility)]
pub struct RedundantImportVisibility {
    #[note]
    pub span: Span,
    #[help]
    pub help: (),

    pub import_vis: String,
    pub max_vis: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unsafe_attr_outside_unsafe)]
pub struct UnsafeAttrOutsideUnsafe {
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: UnsafeAttrOutsideUnsafeSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    lint_unsafe_attr_outside_unsafe_suggestion,
    applicability = "machine-applicable"
)]
pub struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_out_of_scope_macro_calls)]
#[help]
pub struct OutOfScopeMacroCalls {
    pub path: String,
}
