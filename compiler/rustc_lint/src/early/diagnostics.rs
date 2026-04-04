use std::any::Any;
use std::borrow::Cow;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, Level,
    elided_lifetime_in_path_suggestion,
};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;

use crate::lints;

mod check_cfg;

pub struct DiagAndSess<'sess> {
    pub callback: Box<
        dyn for<'b> FnOnce(DiagCtxtHandle<'b>, Level, &dyn Any) -> Diag<'b, ()> + DynSend + 'static,
    >,
    pub sess: &'sess Session,
}

impl<'a> Diagnostic<'a, ()> for DiagAndSess<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        (self.callback)(dcx, level, self.sess)
    }
}

/// This is a diagnostic struct that will decorate a `BuiltinLintDiag`
/// Directly creating the lint structs is expensive, using this will only decorate the lint structs when needed.
pub struct DecorateBuiltinLint<'sess, 'tcx> {
    pub sess: &'sess Session,
    pub tcx: Option<TyCtxt<'tcx>>,
    pub diagnostic: BuiltinLintDiag,
}

impl<'a> Diagnostic<'a, ()> for DecorateBuiltinLint<'_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        match self.diagnostic {
            BuiltinLintDiag::ElidedLifetimesInPaths(
                n,
                path_span,
                incl_angl_brckt,
                insertion_span,
            ) => lints::ElidedLifetimesInPaths {
                subdiag: elided_lifetime_in_path_suggestion(
                    self.sess.source_map(),
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ),
            }
            .into_diag(dcx, level),
            BuiltinLintDiag::UnusedImports {
                remove_whole_use,
                num_to_remove,
                remove_spans,
                test_module_span,
                span_snippets,
            } => {
                let sugg = if remove_whole_use {
                    lints::UnusedImportsSugg::RemoveWholeUse { span: remove_spans[0] }
                } else {
                    lints::UnusedImportsSugg::RemoveImports { remove_spans, num_to_remove }
                };
                let test_module_span =
                    test_module_span.map(|span| self.sess.source_map().guess_head_span(span));

                lints::UnusedImports {
                    sugg,
                    test_module_span,
                    num_snippets: span_snippets.len(),
                    span_snippets: DiagArgValue::StrListSepByAnd(
                        span_snippets.into_iter().map(Cow::Owned).collect(),
                    ),
                }
                .into_diag(dcx, level)
            }
            BuiltinLintDiag::NamedArgumentUsedPositionally {
                position_sp_to_replace,
                position_sp_for_msg,
                named_arg_sp,
                named_arg_name,
                is_formatting_arg,
            } => {
                let (suggestion, name) =
                    if let Some(positional_arg_to_replace) = position_sp_to_replace {
                        let mut name = named_arg_name.clone();
                        if is_formatting_arg {
                            name.push('$')
                        };
                        let span_to_replace = if let Ok(positional_arg_content) =
                            self.sess.source_map().span_to_snippet(positional_arg_to_replace)
                            && positional_arg_content.starts_with(':')
                        {
                            positional_arg_to_replace.shrink_to_lo()
                        } else {
                            positional_arg_to_replace
                        };
                        (Some(span_to_replace), name)
                    } else {
                        (None, String::new())
                    };

                lints::NamedArgumentUsedPositionally {
                    named_arg_sp,
                    position_label_sp: position_sp_for_msg,
                    suggestion,
                    name,
                    named_arg_name,
                }
                .into_diag(dcx, level)
            }

            BuiltinLintDiag::AttributeLint(kind) => {
                DecorateAttrLint { sess: self.sess, tcx: self.tcx, diagnostic: &kind }
                    .into_diag(dcx, level)
            }
        }
    }
}

/// This is a diagnostic struct that will decorate a `AttributeLintKind`
/// Directly creating the lint structs is expensive, using this will only decorate the lint structs when needed.
pub struct DecorateAttrLint<'a, 'sess, 'tcx> {
    pub sess: &'sess Session,
    pub tcx: Option<TyCtxt<'tcx>>,
    pub diagnostic: &'a AttributeLintKind,
}

impl<'a> Diagnostic<'a, ()> for DecorateAttrLint<'_, '_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        match self.diagnostic {
            &AttributeLintKind::UnusedDuplicate { this, other, warning } => {
                lints::UnusedDuplicate { this, other, warning }.into_diag(dcx, level)
            }
            AttributeLintKind::IllFormedAttributeInput { suggestions, docs, help } => {
                lints::IllFormedAttributeInput {
                    num_suggestions: suggestions.len(),
                    suggestions: DiagArgValue::StrListSepByAnd(
                        suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                    ),
                    has_docs: docs.is_some(),
                    docs: docs.unwrap_or(""),
                    help: help.clone().map(|h| lints::IllFormedAttributeInputHelp { lint: h }),
                }
                .into_diag(dcx, level)
            }
            AttributeLintKind::EmptyAttribute { first_span, attr_path, valid_without_list } => {
                lints::EmptyAttributeList {
                    attr_span: *first_span,
                    attr_path: attr_path.clone(),
                    valid_without_list: *valid_without_list,
                }
                .into_diag(dcx, level)
            }
            AttributeLintKind::InvalidTarget { name, target, applied, only, attr_span } => {
                lints::InvalidTargetLint {
                    name: name.clone(),
                    target,
                    applied: DiagArgValue::StrListSepByAnd(
                        applied.into_iter().map(|i| Cow::Owned(i.to_string())).collect(),
                    ),
                    only,
                    attr_span: *attr_span,
                }
                .into_diag(dcx, level)
            }
            &AttributeLintKind::InvalidStyle {
                ref name,
                is_used_as_inner,
                target,
                target_span,
            } => lints::InvalidAttrStyle {
                name: name.clone(),
                is_used_as_inner,
                target_span: (!is_used_as_inner).then_some(target_span),
                target,
            }
            .into_diag(dcx, level),
            &AttributeLintKind::UnsafeAttrOutsideUnsafe { attribute_name_span, sugg_spans } => {
                lints::UnsafeAttrOutsideUnsafeLint {
                    span: attribute_name_span,
                    suggestion: sugg_spans.map(|(left, right)| {
                        lints::UnsafeAttrOutsideUnsafeSuggestion { left, right }
                    }),
                }
                .into_diag(dcx, level)
            }
            &AttributeLintKind::UnexpectedCfgName(name, value) => {
                check_cfg::unexpected_cfg_name(self.sess, self.tcx, name, value)
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::UnexpectedCfgValue(name, value) => {
                check_cfg::unexpected_cfg_value(self.sess, self.tcx, name, value)
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::DuplicateDocAlias { first_definition } => {
                lints::DocAliasDuplicated { first_defn: first_definition }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocAutoCfgExpectsHideOrShow => {
                lints::DocAutoCfgExpectsHideOrShow.into_diag(dcx, level)
            }

            &AttributeLintKind::AmbiguousDeriveHelpers => {
                lints::AmbiguousDeriveHelpers.into_diag(dcx, level)
            }

            &AttributeLintKind::DocAutoCfgHideShowUnexpectedItem { attr_name } => {
                lints::DocAutoCfgHideShowUnexpectedItem { attr_name }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocAutoCfgHideShowExpectsList { attr_name } => {
                lints::DocAutoCfgHideShowExpectsList { attr_name }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocInvalid => lints::DocInvalid.into_diag(dcx, level),

            &AttributeLintKind::DocUnknownInclude { span, inner, value } => {
                lints::DocUnknownInclude {
                    inner,
                    value,
                    sugg: (span, Applicability::MaybeIncorrect),
                }
                .into_diag(dcx, level)
            }

            &AttributeLintKind::DocUnknownSpotlight { span } => {
                lints::DocUnknownSpotlight { sugg_span: span }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocUnknownPasses { name, span } => {
                lints::DocUnknownPasses { name, note_span: span }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocUnknownPlugins { span } => {
                lints::DocUnknownPlugins { label_span: span }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocUnknownAny { name } => {
                lints::DocUnknownAny { name }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocAutoCfgWrongLiteral => {
                lints::DocAutoCfgWrongLiteral.into_diag(dcx, level)
            }

            &AttributeLintKind::DocTestTakesList => lints::DocTestTakesList.into_diag(dcx, level),

            &AttributeLintKind::DocTestUnknown { name } => {
                lints::DocTestUnknown { name }.into_diag(dcx, level)
            }

            &AttributeLintKind::DocTestLiteral => lints::DocTestLiteral.into_diag(dcx, level),

            &AttributeLintKind::AttrCrateLevelOnly => {
                lints::AttrCrateLevelOnly.into_diag(dcx, level)
            }

            &AttributeLintKind::DoNotRecommendDoesNotExpectArgs => {
                lints::DoNotRecommendDoesNotExpectArgs.into_diag(dcx, level)
            }

            &AttributeLintKind::CrateTypeUnknown { span, suggested } => lints::UnknownCrateTypes {
                sugg: suggested.map(|s| lints::UnknownCrateTypesSuggestion { span, snippet: s }),
            }
            .into_diag(dcx, level),

            &AttributeLintKind::MalformedDoc => lints::MalformedDoc.into_diag(dcx, level),

            &AttributeLintKind::ExpectedNoArgs => lints::ExpectedNoArgs.into_diag(dcx, level),

            &AttributeLintKind::ExpectedNameValue => lints::ExpectedNameValue.into_diag(dcx, level),
            &AttributeLintKind::MalformedOnUnimplementedAttr { span } => {
                lints::MalformedOnUnimplementedAttrLint { span }.into_diag(dcx, level)
            }
            &AttributeLintKind::MalformedOnConstAttr { span } => {
                lints::MalformedOnConstAttrLint { span }.into_diag(dcx, level)
            }
            &AttributeLintKind::MalformedOnMissingArgsAttr { span } => {
                lints::MalformedOnMissingArgsAttrLint { span }.into_diag(dcx, level)
            }
            AttributeLintKind::MalformedDiagnosticFormat { warning } => match warning {
                FormatWarning::PositionalArgument { .. } => {
                    lints::DisallowedPositionalArgument.into_diag(dcx, level)
                }
                FormatWarning::InvalidSpecifier { .. } => {
                    lints::InvalidFormatSpecifier.into_diag(dcx, level)
                }
            },
            AttributeLintKind::DiagnosticWrappedParserError { description, label, span } => {
                lints::WrappedParserError { description, label, span: *span }.into_diag(dcx, level)
            }
            &AttributeLintKind::IgnoredDiagnosticOption { option_name, first_span, later_span } => {
                lints::IgnoredDiagnosticOption { option_name, first_span, later_span }
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForOnUnimplemented => {
                lints::MissingOptionsForOnUnimplementedAttr.into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForOnConst => {
                lints::MissingOptionsForOnConstAttr.into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForOnMissingArgs => {
                lints::MissingOptionsForOnMissingArgsAttr.into_diag(dcx, level)
            }
            &AttributeLintKind::MalformedOnMoveAttr { span } => {
                lints::MalformedOnMoveAttrLint { span }.into_diag(dcx, level)
            }
            &AttributeLintKind::OnMoveMalformedFormatLiterals { name } => {
                lints::OnMoveMalformedFormatLiterals { name }.into_diag(dcx, level)
            }
            &AttributeLintKind::OnMoveMalformedAttrExpectedLiteralOrDelimiter => {
                lints::OnMoveMalformedAttrExpectedLiteralOrDelimiter.into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForOnMove => {
                lints::MissingOptionsForOnMoveAttr.into_diag(dcx, level)
            }
            &AttributeLintKind::RenamedLint { name, replace, suggestion } => lints::RenamedLint {
                name,
                replace,
                suggestion: lints::RenamedLintSuggestion::WithSpan { suggestion, replace },
            }
            .into_diag(dcx, level),
            &AttributeLintKind::DeprecatedLintName { name, suggestion, replace } => {
                lints::DeprecatedLintName { name, suggestion, replace }.into_diag(dcx, level)
            }
            &AttributeLintKind::RemovedLint { name, ref reason } => {
                lints::RemovedLint { name, reason }.into_diag(dcx, level)
            }
            &AttributeLintKind::UnknownLint { name, span, suggestion } => lints::UnknownLint {
                name,
                suggestion: suggestion.map(|(replace, from_rustc)| {
                    lints::UnknownLintSuggestion::WithSpan { suggestion: span, replace, from_rustc }
                }),
            }
            .into_diag(dcx, level),
            &AttributeLintKind::IgnoredUnlessCrateSpecified { level: attr_level, name } => {
                lints::IgnoredUnlessCrateSpecified { level: attr_level, name }.into_diag(dcx, level)
            }
        }
    }
}
