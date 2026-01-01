use std::borrow::Cow;

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, Level,
    elided_lifetime_in_path_suggestion,
};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
use rustc_middle::middle::stability;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_span::BytePos;
use tracing::debug;

use crate::lints;

mod check_cfg;

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
            BuiltinLintDiag::UnicodeTextFlow(comment_span, content) => {
                let spans: Vec<_> = content
                    .char_indices()
                    .filter_map(|(i, c)| {
                        TEXT_FLOW_CONTROL_CHARS.contains(&c).then(|| {
                            let lo = comment_span.lo() + BytePos(2 + i as u32);
                            (c, comment_span.with_lo(lo).with_hi(lo + BytePos(c.len_utf8() as u32)))
                        })
                    })
                    .collect();
                let characters = spans
                    .iter()
                    .map(|&(c, span)| lints::UnicodeCharNoteSub { span, c_debug: format!("{c:?}") })
                    .collect();
                let suggestions = (!spans.is_empty()).then_some(lints::UnicodeTextFlowSuggestion {
                    spans: spans.iter().map(|(_c, span)| *span).collect(),
                });

                lints::UnicodeTextFlow {
                    comment_span,
                    characters,
                    suggestions,
                    num_codepoints: spans.len(),
                }
                .into_diag(dcx, level)
            }
            BuiltinLintDiag::AbsPathWithModule(mod_span) => {
                let (replacement, applicability) =
                    match self.sess.source_map().span_to_snippet(mod_span) {
                        Ok(ref s) => {
                            // FIXME(Manishearth) ideally the emitting code
                            // can tell us whether or not this is global
                            let opt_colon =
                                if s.trim_start().starts_with("::") { "" } else { "::" };

                            (format!("crate{opt_colon}{s}"), Applicability::MachineApplicable)
                        }
                        Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders),
                    };
                lints::AbsPathWithModule {
                    sugg: lints::AbsPathWithModuleSugg {
                        span: mod_span,
                        applicability,
                        replacement,
                    },
                }
                .into_diag(dcx, level)
            }
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
            BuiltinLintDiag::RedundantImport(spans, ident) => {
                let subs = spans
                    .into_iter()
                    .map(|(span, is_imported)| match (span.is_dummy(), is_imported) {
                        (false, true) => lints::RedundantImportSub::ImportedHere { span, ident },
                        (false, false) => lints::RedundantImportSub::DefinedHere { span, ident },
                        (true, true) => lints::RedundantImportSub::ImportedPrelude { span, ident },
                        (true, false) => lints::RedundantImportSub::DefinedPrelude { span, ident },
                    })
                    .collect();
                lints::RedundantImport { subs, ident }.into_diag(dcx, level)
            }
            BuiltinLintDiag::DeprecatedMacro {
                suggestion,
                suggestion_span,
                note,
                path,
                since_kind,
            } => {
                let sub = suggestion.map(|suggestion| stability::DeprecationSuggestion {
                    span: suggestion_span,
                    kind: "macro".to_owned(),
                    suggestion,
                });

                stability::Deprecated { sub, kind: "macro".to_owned(), path, note, since_kind }
                    .into_diag(dcx, level)
            }
            BuiltinLintDiag::PatternsInFnsWithoutBody { span: remove_span, ident, is_foreign } => {
                let sub = lints::PatternsInFnsWithoutBodySub { ident, span: remove_span };
                if is_foreign {
                    lints::PatternsInFnsWithoutBody::Foreign { sub }
                } else {
                    lints::PatternsInFnsWithoutBody::Bodiless { sub }
                }
                .into_diag(dcx, level)
            }
            BuiltinLintDiag::ReservedPrefix(label_span, prefix) => lints::ReservedPrefix {
                label: label_span,
                suggestion: label_span.shrink_to_hi(),
                prefix,
            }
            .into_diag(dcx, level),
            BuiltinLintDiag::RawPrefix(label_span) => {
                lints::RawPrefix { label: label_span, suggestion: label_span.shrink_to_hi() }
                    .into_diag(dcx, level)
            }
            BuiltinLintDiag::ReservedString { is_string, suggestion } => {
                if is_string {
                    lints::ReservedString { suggestion }.into_diag(dcx, level)
                } else {
                    lints::ReservedMultihash { suggestion }.into_diag(dcx, level)
                }
            }
            BuiltinLintDiag::BreakWithLabelAndLoop(sugg_span) => lints::BreakWithLabelAndLoop {
                sub: lints::BreakWithLabelAndLoopSub {
                    left: sugg_span.shrink_to_lo(),
                    right: sugg_span.shrink_to_hi(),
                },
            }
            .into_diag(dcx, level),
            BuiltinLintDiag::DeprecatedWhereclauseLocation(left_sp, sugg) => {
                let suggestion = match sugg {
                    Some((right_sp, sugg)) => lints::DeprecatedWhereClauseLocationSugg::MoveToEnd {
                        left: left_sp,
                        right: right_sp,
                        sugg,
                    },
                    None => lints::DeprecatedWhereClauseLocationSugg::RemoveWhere { span: left_sp },
                };
                lints::DeprecatedWhereClauseLocation { suggestion }.into_diag(dcx, level)
            }
            BuiltinLintDiag::SingleUseLifetime {
                param_span,
                use_span: Some((use_span, elide)),
                deletion_span,
                ident,
            } => {
                debug!(?param_span, ?use_span, ?deletion_span);
                let suggestion = if let Some(deletion_span) = deletion_span {
                    let (use_span, replace_lt) = if elide {
                        let use_span =
                            self.sess.source_map().span_extend_while_whitespace(use_span);
                        (use_span, String::new())
                    } else {
                        (use_span, "'_".to_owned())
                    };
                    debug!(?deletion_span, ?use_span);

                    // issue 107998 for the case such as a wrong function pointer type
                    // `deletion_span` is empty and there is no need to report lifetime uses here
                    let deletion_span =
                        if deletion_span.is_empty() { None } else { Some(deletion_span) };
                    Some(lints::SingleUseLifetimeSugg { deletion_span, use_span, replace_lt })
                } else {
                    None
                };

                lints::SingleUseLifetime { suggestion, param_span, use_span, ident }
                    .into_diag(dcx, level)
            }
            BuiltinLintDiag::SingleUseLifetime { use_span: None, deletion_span, ident, .. } => {
                lints::UnusedLifetime { deletion_span, ident }.into_diag(dcx, level)
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
            BuiltinLintDiag::AmbiguousGlobReexports {
                name,
                namespace,
                first_reexport_span,
                duplicate_reexport_span,
            } => lints::AmbiguousGlobReexports {
                first_reexport: first_reexport_span,
                duplicate_reexport: duplicate_reexport_span,
                name,
                namespace,
            }
            .into_diag(dcx, level),
            BuiltinLintDiag::HiddenGlobReexports {
                name,
                namespace,
                glob_reexport_span,
                private_item_span,
            } => lints::HiddenGlobReexports {
                glob_reexport: glob_reexport_span,
                private_item: private_item_span,

                name,
                namespace,
            }
            .into_diag(dcx, level),
            BuiltinLintDiag::UnusedQualifications { removal_span } => {
                lints::UnusedQualifications { removal_span }.into_diag(dcx, level)
            }
            BuiltinLintDiag::AssociatedConstElidedLifetime {
                elided,
                span: lt_span,
                lifetimes_in_scope,
            } => {
                let lt_span = if elided { lt_span.shrink_to_hi() } else { lt_span };
                let code = if elided { "'static " } else { "'static" };
                lints::AssociatedConstElidedLifetime {
                    span: lt_span,
                    code,
                    elided,
                    lifetimes_in_scope,
                }
                .into_diag(dcx, level)
            }
            BuiltinLintDiag::UnreachableCfg { span, wildcard_span } => match wildcard_span {
                Some(wildcard_span) => {
                    lints::UnreachableCfgSelectPredicateWildcard { span, wildcard_span }
                        .into_diag(dcx, level)
                }
                None => lints::UnreachableCfgSelectPredicate { span }.into_diag(dcx, level),
            },

            BuiltinLintDiag::UnusedCrateDependency { extern_crate, local_crate } => {
                lints::UnusedCrateDependency { extern_crate, local_crate }.into_diag(dcx, level)
            }
            BuiltinLintDiag::UnusedVisibility(span) => {
                lints::UnusedVisibility { span }.into_diag(dcx, level)
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
            AttributeLintKind::IllFormedAttributeInput { suggestions, docs } => {
                lints::IllFormedAttributeInput {
                    num_suggestions: suggestions.len(),
                    suggestions: DiagArgValue::StrListSepByAnd(
                        suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                    ),
                    has_docs: docs.is_some(),
                    docs: docs.unwrap_or(""),
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
        }
    }
}
