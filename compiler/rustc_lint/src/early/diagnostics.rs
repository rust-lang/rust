use std::borrow::Cow;

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, Diagnostic, elided_lifetime_in_path_suggestion,
};
use rustc_hir::lints::AttributeLintKind;
use rustc_middle::middle::stability;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_span::BytePos;
use tracing::debug;

use crate::lints;

mod check_cfg;

pub fn decorate_builtin_lint(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    diagnostic: BuiltinLintDiag,
    diag: &mut Diag<'_, ()>,
) {
    match diagnostic {
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

            let diag2: Diag<'_, ()> = lints::UnicodeTextFlow {
                comment_span,
                characters,
                suggestions,
                num_codepoints: spans.len(),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::AbsPathWithModule(mod_span) => {
            let (replacement, applicability) = match sess.source_map().span_to_snippet(mod_span) {
                Ok(ref s) => {
                    // FIXME(Manishearth) ideally the emitting code
                    // can tell us whether or not this is global
                    let opt_colon = if s.trim_start().starts_with("::") { "" } else { "::" };

                    (format!("crate{opt_colon}{s}"), Applicability::MachineApplicable)
                }
                Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders),
            };
            let diag2: Diag<'_, ()> = lints::AbsPathWithModule {
                sugg: lints::AbsPathWithModuleSugg { span: mod_span, applicability, replacement },
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(n, path_span, incl_angl_brckt, insertion_span) => {
            let diag2: Diag<'_, ()> = lints::ElidedLifetimesInPaths {
                subdiag: elided_lifetime_in_path_suggestion(
                    sess.source_map(),
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
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
                test_module_span.map(|span| sess.source_map().guess_head_span(span));

            let diag2: Diag<'_, ()> = lints::UnusedImports {
                sugg,
                test_module_span,
                num_snippets: span_snippets.len(),
                span_snippets: DiagArgValue::StrListSepByAnd(
                    span_snippets.into_iter().map(Cow::Owned).collect(),
                ),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::RedundantImport(spans, ident) => {
            let subs = spans
                .into_iter()
                .map(|(span, is_imported)| {
                    (match (span.is_dummy(), is_imported) {
                        (false, true) => lints::RedundantImportSub::ImportedHere,
                        (false, false) => lints::RedundantImportSub::DefinedHere,
                        (true, true) => lints::RedundantImportSub::ImportedPrelude,
                        (true, false) => lints::RedundantImportSub::DefinedPrelude,
                    })(span)
                })
                .collect();
            let diag2: Diag<'_, ()> =
                lints::RedundantImport { subs, ident }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
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

            let diag2: Diag<'_, ()> =
                stability::Deprecated { sub, kind: "macro".to_owned(), path, note, since_kind }
                    .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody { span: remove_span, ident, is_foreign } => {
            let sub = lints::PatternsInFnsWithoutBodySub { ident, span: remove_span };
            let diag2: Diag<'_, ()> = if is_foreign {
                lints::PatternsInFnsWithoutBody::Foreign { sub }
            } else {
                lints::PatternsInFnsWithoutBody::Bodiless { sub }
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::ReservedPrefix(label_span, prefix) => {
            let diag2: Diag<'_, ()> = lints::ReservedPrefix {
                label: label_span,
                suggestion: label_span.shrink_to_hi(),
                prefix,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::RawPrefix(label_span) => {
            let diag2: Diag<'_, ()> =
                lints::RawPrefix { label: label_span, suggestion: label_span.shrink_to_hi() }
                    .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::ReservedString { is_string, suggestion } => {
            let diag2: Diag<'_, ()> = if is_string {
                lints::ReservedString { suggestion }.into_diag(diag.dcx, diag.level())
            } else {
                lints::ReservedMultihash { suggestion }.into_diag(diag.dcx, diag.level())
            };
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::BreakWithLabelAndLoop(sugg_span) => {
            let diag2: Diag<'_, ()> = lints::BreakWithLabelAndLoop {
                sub: lints::BreakWithLabelAndLoopSub {
                    left: sugg_span.shrink_to_lo(),
                    right: sugg_span.shrink_to_hi(),
                },
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::DeprecatedWhereclauseLocation(left_sp, sugg) => {
            let suggestion = match sugg {
                Some((right_sp, sugg)) => lints::DeprecatedWhereClauseLocationSugg::MoveToEnd {
                    left: left_sp,
                    right: right_sp,
                    sugg,
                },
                None => lints::DeprecatedWhereClauseLocationSugg::RemoveWhere { span: left_sp },
            };
            let diag2: Diag<'_, ()> = lints::DeprecatedWhereClauseLocation { suggestion }
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
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
                    let use_span = sess.source_map().span_extend_while_whitespace(use_span);
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

            let diag2: Diag<'_, ()> =
                lints::SingleUseLifetime { suggestion, param_span, use_span, ident }
                    .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::SingleUseLifetime { use_span: None, deletion_span, ident, .. } => {
            let diag2: Diag<'_, ()> =
                lints::UnusedLifetime { deletion_span, ident }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::NamedArgumentUsedPositionally {
            position_sp_to_replace,
            position_sp_for_msg,
            named_arg_sp,
            named_arg_name,
            is_formatting_arg,
        } => {
            let (suggestion, name) = if let Some(positional_arg_to_replace) = position_sp_to_replace
            {
                let mut name = named_arg_name.clone();
                if is_formatting_arg {
                    name.push('$')
                };
                let span_to_replace = if let Ok(positional_arg_content) =
                    sess.source_map().span_to_snippet(positional_arg_to_replace)
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

            let diag2: Diag<'_, ()> = lints::NamedArgumentUsedPositionally {
                named_arg_sp,
                position_label_sp: position_sp_for_msg,
                suggestion,
                name,
                named_arg_name,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::AmbiguousGlobReexports {
            name,
            namespace,
            first_reexport_span,
            duplicate_reexport_span,
        } => {
            let diag2: Diag<'_, ()> = lints::AmbiguousGlobReexports {
                first_reexport: first_reexport_span,
                duplicate_reexport: duplicate_reexport_span,
                name,
                namespace,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::HiddenGlobReexports {
            name,
            namespace,
            glob_reexport_span,
            private_item_span,
        } => {
            let diag2: Diag<'_, ()> = lints::HiddenGlobReexports {
                glob_reexport: glob_reexport_span,
                private_item: private_item_span,

                name,
                namespace,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::UnusedQualifications { removal_span } => {
            let diag2: Diag<'_, ()> =
                lints::UnusedQualifications { removal_span }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::AssociatedConstElidedLifetime {
            elided,
            span: lt_span,
            lifetimes_in_scope,
        } => {
            let lt_span = if elided { lt_span.shrink_to_hi() } else { lt_span };
            let code = if elided { "'static " } else { "'static" };
            let diag2: Diag<'_, ()> = lints::AssociatedConstElidedLifetime {
                span: lt_span,
                code,
                elided,
                lifetimes_in_scope,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::UnreachableCfg { span, wildcard_span } => match wildcard_span {
            Some(wildcard_span) => {
                let diag2: Diag<'_, ()> =
                    lints::UnreachableCfgSelectPredicateWildcard { span, wildcard_span }
                        .into_diag(diag.dcx, diag.level());
                diag.merge_with_other_diag(diag2);
            }
            None => {
                let diag2: Diag<'_, ()> =
                    lints::UnreachableCfgSelectPredicate { span }.into_diag(diag.dcx, diag.level());
                diag.merge_with_other_diag(diag2);
            }
        },

        BuiltinLintDiag::UnusedCrateDependency { extern_crate, local_crate } => {
            let diag2: Diag<'_, ()> = lints::UnusedCrateDependency { extern_crate, local_crate }
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::UnusedVisibility(span) => {
            let diag2: Diag<'_, ()> =
                lints::UnusedVisibility { span }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        BuiltinLintDiag::AttributeLint(kind) => decorate_attribute_lint(sess, tcx, &kind, diag),
    }
}

pub fn decorate_attribute_lint(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    kind: &AttributeLintKind,
    diag: &mut Diag<'_, ()>,
) {
    match kind {
        &AttributeLintKind::UnusedDuplicate { this, other, warning } => {
            let diag2: Diag<'_, ()> =
                lints::UnusedDuplicate { this, other, warning }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        AttributeLintKind::IllFormedAttributeInput { suggestions, docs } => {
            let diag2: Diag<'_, ()> = lints::IllFormedAttributeInput {
                num_suggestions: suggestions.len(),
                suggestions: DiagArgValue::StrListSepByAnd(
                    suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                ),
                has_docs: docs.is_some(),
                docs: docs.unwrap_or(""),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        AttributeLintKind::EmptyAttribute { first_span, attr_path, valid_without_list } => {
            let diag2: Diag<'_, ()> = lints::EmptyAttributeList {
                attr_span: *first_span,
                attr_path: attr_path.clone(),
                valid_without_list: *valid_without_list,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        AttributeLintKind::InvalidTarget { name, target, applied, only, attr_span } => {
            let diag2: Diag<'_, ()> = lints::InvalidTargetLint {
                name: name.clone(),
                target,
                applied: DiagArgValue::StrListSepByAnd(
                    applied.into_iter().map(|i| Cow::Owned(i.to_string())).collect(),
                ),
                only,
                attr_span: *attr_span,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        &AttributeLintKind::InvalidStyle { ref name, is_used_as_inner, target, target_span } => {
            let diag2: Diag<'_, ()> = lints::InvalidAttrStyle {
                name: name.clone(),
                is_used_as_inner,
                target_span: (!is_used_as_inner).then_some(target_span),
                target,
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        &AttributeLintKind::UnsafeAttrOutsideUnsafe { attribute_name_span, sugg_spans } => {
            let diag2: Diag<'_, ()> = lints::UnsafeAttrOutsideUnsafeLint {
                span: attribute_name_span,
                suggestion: sugg_spans
                    .map(|(left, right)| lints::UnsafeAttrOutsideUnsafeSuggestion { left, right }),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        &AttributeLintKind::UnexpectedCfgName(name, value) => {
            let diag2: Diag<'_, ()> = check_cfg::unexpected_cfg_name(sess, tcx, name, value)
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        &AttributeLintKind::UnexpectedCfgValue(name, value) => {
            let diag2: Diag<'_, ()> = check_cfg::unexpected_cfg_value(sess, tcx, name, value)
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
        &AttributeLintKind::DuplicateDocAlias { first_definition } => {
            let diag2: Diag<'_, ()> = lints::DocAliasDuplicated { first_defn: first_definition }
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocAutoCfgExpectsHideOrShow => {
            let diag2: Diag<'_, ()> =
                lints::DocAutoCfgExpectsHideOrShow.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::AmbiguousDeriveHelpers => {
            let diag2: Diag<'_, ()> =
                lints::AmbiguousDeriveHelpers.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocAutoCfgHideShowUnexpectedItem { attr_name } => {
            let diag2: Diag<'_, ()> = lints::DocAutoCfgHideShowUnexpectedItem { attr_name }
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocAutoCfgHideShowExpectsList { attr_name } => {
            let diag2: Diag<'_, ()> = lints::DocAutoCfgHideShowExpectsList { attr_name }
                .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocInvalid => {
            let diag2: Diag<'_, ()> = lints::DocInvalid.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocUnknownInclude { span, inner, value } => {
            let diag2: Diag<'_, ()> = lints::DocUnknownInclude {
                inner,
                value,
                sugg: (span, Applicability::MaybeIncorrect),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocUnknownSpotlight { span } => {
            let diag2: Diag<'_, ()> =
                lints::DocUnknownSpotlight { sugg_span: span }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocUnknownPasses { name, span } => {
            let diag2: Diag<'_, ()> =
                lints::DocUnknownPasses { name, note_span: span }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocUnknownPlugins { span } => {
            let diag2: Diag<'_, ()> =
                lints::DocUnknownPlugins { label_span: span }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocUnknownAny { name } => {
            let diag2: Diag<'_, ()> =
                lints::DocUnknownAny { name }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocAutoCfgWrongLiteral => {
            let diag2: Diag<'_, ()> =
                lints::DocAutoCfgWrongLiteral.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocTestTakesList => {
            let diag2: Diag<'_, ()> = lints::DocTestTakesList.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocTestUnknown { name } => {
            let diag2: Diag<'_, ()> =
                lints::DocTestUnknown { name }.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DocTestLiteral => {
            let diag2: Diag<'_, ()> = lints::DocTestLiteral.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::AttrCrateLevelOnly => {
            let diag2: Diag<'_, ()> = lints::AttrCrateLevelOnly.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::DoNotRecommendDoesNotExpectArgs => {
            let diag2: Diag<'_, ()> =
                lints::DoNotRecommendDoesNotExpectArgs.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::CrateTypeUnknown { span, suggested } => {
            let diag2: Diag<'_, ()> = lints::UnknownCrateTypes {
                sugg: suggested.map(|s| lints::UnknownCrateTypesSuggestion { span, snippet: s }),
            }
            .into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::MalformedDoc => {
            let diag2: Diag<'_, ()> = lints::MalformedDoc.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::ExpectedNoArgs => {
            let diag2: Diag<'_, ()> = lints::ExpectedNoArgs.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }

        &AttributeLintKind::ExpectedNameValue => {
            let diag2: Diag<'_, ()> = lints::ExpectedNameValue.into_diag(diag.dcx, diag.level());
            diag.merge_with_other_diag(diag2);
        }
    }
}
