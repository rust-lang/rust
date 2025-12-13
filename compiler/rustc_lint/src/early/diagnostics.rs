use std::borrow::Cow;

use rustc_errors::{
    Applicability, Diag, DiagArgValue, LintDiagnostic, elided_lifetime_in_path_suggestion,
};
use rustc_hir::lints::AttributeLintKind;
use rustc_middle::middle::stability;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
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
            lints::AbsPathWithModule {
                sugg: lints::AbsPathWithModuleSugg { span: mod_span, applicability, replacement },
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(n, path_span, incl_angl_brckt, insertion_span) => {
            lints::ElidedLifetimesInPaths {
                subdiag: elided_lifetime_in_path_suggestion(
                    sess.source_map(),
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ),
            }
            .decorate_lint(diag);
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

            lints::UnusedImports {
                sugg,
                test_module_span,
                num_snippets: span_snippets.len(),
                span_snippets: DiagArgValue::StrListSepByAnd(
                    span_snippets.into_iter().map(Cow::Owned).collect(),
                ),
            }
            .decorate_lint(diag);
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
            lints::RedundantImport { subs, ident }.decorate_lint(diag);
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
                .decorate_lint(diag);
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

            lints::SingleUseLifetime { suggestion, param_span, use_span, ident }
                .decorate_lint(diag);
        }
        BuiltinLintDiag::SingleUseLifetime { use_span: None, deletion_span, ident, .. } => {
            lints::UnusedLifetime { deletion_span, ident }.decorate_lint(diag);
        }
        BuiltinLintDiag::AmbiguousGlobReexports {
            name,
            namespace,
            first_reexport_span,
            duplicate_reexport_span,
        } => {
            lints::AmbiguousGlobReexports {
                first_reexport: first_reexport_span,
                duplicate_reexport: duplicate_reexport_span,
                name,
                namespace,
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::HiddenGlobReexports {
            name,
            namespace,
            glob_reexport_span,
            private_item_span,
        } => {
            lints::HiddenGlobReexports {
                glob_reexport: glob_reexport_span,
                private_item: private_item_span,

                name,
                namespace,
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedQualifications { removal_span } => {
            lints::UnusedQualifications { removal_span }.decorate_lint(diag);
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
            .decorate_lint(diag);
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
            lints::UnusedDuplicate { this, other, warning }.decorate_lint(diag)
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
            .decorate_lint(diag)
        }
        AttributeLintKind::EmptyAttribute { first_span, attr_path, valid_without_list } => {
            lints::EmptyAttributeList {
                attr_span: *first_span,
                attr_path: attr_path.clone(),
                valid_without_list: *valid_without_list,
            }
            .decorate_lint(diag)
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
            .decorate_lint(diag)
        }
        &AttributeLintKind::InvalidStyle { ref name, is_used_as_inner, target, target_span } => {
            lints::InvalidAttrStyle {
                name: name.clone(),
                is_used_as_inner,
                target_span: (!is_used_as_inner).then_some(target_span),
                target,
            }
            .decorate_lint(diag)
        }
        &AttributeLintKind::UnsafeAttrOutsideUnsafe { attribute_name_span, sugg_spans } => {
            lints::UnsafeAttrOutsideUnsafeLint {
                span: attribute_name_span,
                suggestion: sugg_spans
                    .map(|(left, right)| lints::UnsafeAttrOutsideUnsafeSuggestion { left, right }),
            }
            .decorate_lint(diag)
        }
        &AttributeLintKind::UnexpectedCfgName(name, value) => {
            check_cfg::unexpected_cfg_name(sess, tcx, name, value).decorate_lint(diag)
        }
        &AttributeLintKind::UnexpectedCfgValue(name, value) => {
            check_cfg::unexpected_cfg_value(sess, tcx, name, value).decorate_lint(diag)
        }
        &AttributeLintKind::DuplicateDocAlias { first_definition } => {
            lints::DocAliasDuplicated { first_defn: first_definition }.decorate_lint(diag)
        }

        &AttributeLintKind::DocAutoCfgExpectsHideOrShow => {
            lints::DocAutoCfgExpectsHideOrShow.decorate_lint(diag)
        }

        &AttributeLintKind::DocAutoCfgHideShowUnexpectedItem { attr_name } => {
            lints::DocAutoCfgHideShowUnexpectedItem { attr_name }.decorate_lint(diag)
        }

        &AttributeLintKind::DocAutoCfgHideShowExpectsList { attr_name } => {
            lints::DocAutoCfgHideShowExpectsList { attr_name }.decorate_lint(diag)
        }

        &AttributeLintKind::DocInvalid => { lints::DocInvalid }.decorate_lint(diag),

        &AttributeLintKind::DocUnknownInclude { span, inner, value } => {
            lints::DocUnknownInclude { inner, value, sugg: (span, Applicability::MaybeIncorrect) }
        }
        .decorate_lint(diag),

        &AttributeLintKind::DocUnknownSpotlight { span } => {
            lints::DocUnknownSpotlight { sugg_span: span }.decorate_lint(diag)
        }

        &AttributeLintKind::DocUnknownPasses { name, span } => {
            lints::DocUnknownPasses { name, note_span: span }.decorate_lint(diag)
        }

        &AttributeLintKind::DocUnknownPlugins { span } => {
            lints::DocUnknownPlugins { label_span: span }.decorate_lint(diag)
        }

        &AttributeLintKind::DocUnknownAny { name } => {
            lints::DocUnknownAny { name }.decorate_lint(diag)
        }

        &AttributeLintKind::DocAutoCfgWrongLiteral => {
            lints::DocAutoCfgWrongLiteral.decorate_lint(diag)
        }

        &AttributeLintKind::DocTestTakesList => lints::DocTestTakesList.decorate_lint(diag),

        &AttributeLintKind::DocTestUnknown { name } => {
            lints::DocTestUnknown { name }.decorate_lint(diag)
        }

        &AttributeLintKind::DocTestLiteral => lints::DocTestLiteral.decorate_lint(diag),
    }
}
