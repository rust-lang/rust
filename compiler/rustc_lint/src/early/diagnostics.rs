use std::any::Any;
use std::borrow::Cow;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, Level};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

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
            &AttributeLintKind::MalFormedDiagnosticAttribute { attribute, span } => {
                lints::MalFormedDiagnosticAttributeLint { attribute, span }.into_diag(dcx, level)
            }

            AttributeLintKind::MalformedDiagnosticFormat { warning } => match warning {
                FormatWarning::PositionalArgument { .. } => {
                    lints::DisallowedPositionalArgument.into_diag(dcx, level)
                }
                FormatWarning::InvalidSpecifier { .. } => {
                    lints::InvalidFormatSpecifier.into_diag(dcx, level)
                }
                FormatWarning::DisallowedPlaceholder { .. } => {
                    lints::DisallowedPlaceholder.into_diag(dcx, level)
                }
            },
            AttributeLintKind::DiagnosticWrappedParserError { description, label, span } => {
                lints::WrappedParserError { description, label, span: *span }.into_diag(dcx, level)
            }
            &AttributeLintKind::IgnoredDiagnosticOption { option_name, first_span, later_span } => {
                lints::IgnoredDiagnosticOption { option_name, first_span, later_span }
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForDiagnosticAttribute { attribute } => {
                lints::MissingOptionsForDiagnosticAttribute { attribute }.into_diag(dcx, level)
            }
            &AttributeLintKind::OnMoveMalformedFormatLiterals { name } => {
                lints::OnMoveMalformedFormatLiterals { name }.into_diag(dcx, level)
            }
            &AttributeLintKind::NonMetaItemDiagnosticAttribute => {
                lints::NonMetaItemDiagnosticAttribute.into_diag(dcx, level)
            }
        }
    }
}
