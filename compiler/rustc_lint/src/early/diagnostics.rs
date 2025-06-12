#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::borrow::Cow;

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, LintDiagnostic, elided_lifetime_in_path_suggestion,
};
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

            lints::UnicodeTextFlow {
                comment_span,
                characters,
                suggestions,
                num_codepoints: spans.len(),
            }
            .decorate_lint(diag);
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
            lints::AbsPathWithModule {
                sugg: lints::AbsPathWithModuleSugg { span: mod_span, applicability, replacement },
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::ProcMacroDeriveResolutionFallback { span: macro_span, ns, ident } => {
            lints::ProcMacroDeriveResolutionFallback { span: macro_span, ns, ident }
                .decorate_lint(diag)
        }
        BuiltinLintDiag::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def) => {
            lints::MacroExpandedMacroExportsAccessedByAbsolutePaths { definition: span_def }
                .decorate_lint(diag)
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
        BuiltinLintDiag::UnknownCrateTypes { span, candidate } => {
            let sugg = candidate.map(|candidate| lints::UnknownCrateTypesSub { span, candidate });
            lints::UnknownCrateTypes { sugg }.decorate_lint(diag);
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
        BuiltinLintDiag::UnusedDocComment(attr_span) => {
            lints::UnusedDocComment { span: attr_span }.decorate_lint(diag);
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody { span: remove_span, ident, is_foreign } => {
            let sub = lints::PatternsInFnsWithoutBodySub { ident, span: remove_span };
            if is_foreign {
                lints::PatternsInFnsWithoutBody::Foreign { sub }
            } else {
                lints::PatternsInFnsWithoutBody::Bodiless { sub }
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::MissingAbi(label_span, default_abi) => {
            lints::MissingAbi { span: label_span, default_abi }.decorate_lint(diag);
        }
        BuiltinLintDiag::LegacyDeriveHelpers(label_span) => {
            lints::LegacyDeriveHelpers { span: label_span }.decorate_lint(diag);
        }
        BuiltinLintDiag::OrPatternsBackCompat(suggestion_span, suggestion) => {
            lints::OrPatternsBackCompat { span: suggestion_span, suggestion }.decorate_lint(diag);
        }
        BuiltinLintDiag::ReservedPrefix(label_span, prefix) => {
            lints::ReservedPrefix {
                label: label_span,
                suggestion: label_span.shrink_to_hi(),
                prefix,
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::RawPrefix(label_span) => {
            lints::RawPrefix { label: label_span, suggestion: label_span.shrink_to_hi() }
                .decorate_lint(diag);
        }
        BuiltinLintDiag::ReservedString { is_string, suggestion } => {
            if is_string {
                lints::ReservedString { suggestion }.decorate_lint(diag);
            } else {
                lints::ReservedMultihash { suggestion }.decorate_lint(diag);
            }
        }
        BuiltinLintDiag::HiddenUnicodeCodepoints {
            label,
            count,
            span_label,
            labels,
            escape,
            spans,
        } => {
            lints::HiddenUnicodeCodepointsDiag {
                label: &label,
                count,
                span_label,
                labels: labels.map(|spans| lints::HiddenUnicodeCodepointsDiagLabels { spans }),
                sub: if escape {
                    lints::HiddenUnicodeCodepointsDiagSub::Escape { spans }
                } else {
                    lints::HiddenUnicodeCodepointsDiagSub::NoEscape { spans }
                },
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedBuiltinAttribute { attr_name, macro_name, invoc_span } => {
            lints::UnusedBuiltinAttribute { invoc_span, attr_name, macro_name }.decorate_lint(diag);
        }
        BuiltinLintDiag::TrailingMacro(is_trailing, name) => {
            lints::TrailingMacro { is_trailing, name }.decorate_lint(diag);
        }
        BuiltinLintDiag::BreakWithLabelAndLoop(sugg_span) => {
            lints::BreakWithLabelAndLoop {
                sub: lints::BreakWithLabelAndLoopSub {
                    left: sugg_span.shrink_to_lo(),
                    right: sugg_span.shrink_to_hi(),
                },
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::UnexpectedCfgName(name, value) => {
            check_cfg::unexpected_cfg_name(sess, tcx, name, value).decorate_lint(diag);
        }
        BuiltinLintDiag::UnexpectedCfgValue(name, value) => {
            check_cfg::unexpected_cfg_value(sess, tcx, name, value).decorate_lint(diag);
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
            lints::DeprecatedWhereClauseLocation { suggestion }.decorate_lint(diag);
        }
        BuiltinLintDiag::MissingUnsafeOnExtern { suggestion } => {
            lints::MissingUnsafeOnExtern { suggestion }.decorate_lint(diag);
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
            debug!(?deletion_span);
            lints::UnusedLifetime { deletion_span, ident }.decorate_lint(diag);
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

            lints::NamedArgumentUsedPositionally {
                named_arg_sp,
                position_label_sp: position_sp_for_msg,
                suggestion,
                name,
                named_arg_name,
            }
            .decorate_lint(diag);
        }
        BuiltinLintDiag::ByteSliceInPackedStructWithDerive { ty } => {
            lints::ByteSliceInPackedStructWithDerive { ty }.decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedExternCrate { span, removal_span } => {
            lints::UnusedExternCrate { span, removal_span }.decorate_lint(diag);
        }
        BuiltinLintDiag::ExternCrateNotIdiomatic { vis_span, ident_span } => {
            let suggestion_span = vis_span.between(ident_span);
            let code = if vis_span.is_empty() { "use " } else { " use " };

            lints::ExternCrateNotIdiomatic { span: suggestion_span, code }.decorate_lint(diag);
        }
        BuiltinLintDiag::AmbiguousGlobImports { diag: ambiguity } => {
            lints::AmbiguousGlobImports { ambiguity }.decorate_lint(diag);
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
        BuiltinLintDiag::UnsafeAttrOutsideUnsafe {
            attribute_name_span,
            sugg_spans: (left, right),
        } => {
            lints::UnsafeAttrOutsideUnsafe {
                span: attribute_name_span,
                suggestion: lints::UnsafeAttrOutsideUnsafeSuggestion { left, right },
            }
            .decorate_lint(diag);
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
        BuiltinLintDiag::RedundantImportVisibility { max_vis, span: vis_span, import_vis } => {
            lints::RedundantImportVisibility { span: vis_span, help: (), max_vis, import_vis }
                .decorate_lint(diag);
        }
        BuiltinLintDiag::UnknownDiagnosticAttribute { span: typo_span, typo_name } => {
            let typo = typo_name.map(|typo_name| lints::UnknownDiagnosticAttributeTypoSugg {
                span: typo_span,
                typo_name,
            });
            lints::UnknownDiagnosticAttribute { typo }.decorate_lint(diag);
        }
        BuiltinLintDiag::MacroUseDeprecated => {
            lints::MacroUseDeprecated.decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedMacroUse => lints::UnusedMacroUse.decorate_lint(diag),
        BuiltinLintDiag::PrivateExternCrateReexport { source: ident, extern_crate_span } => {
            lints::PrivateExternCrateReexport { ident, sugg: extern_crate_span.shrink_to_lo() }
                .decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedLabel => lints::UnusedLabel.decorate_lint(diag),
        BuiltinLintDiag::MacroIsPrivate(ident) => {
            lints::MacroIsPrivate { ident }.decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedMacroDefinition(name) => {
            lints::UnusedMacroDefinition { name }.decorate_lint(diag);
        }
        BuiltinLintDiag::MacroRuleNeverUsed(n, name) => {
            lints::MacroRuleNeverUsed { n: n + 1, name }.decorate_lint(diag);
        }
        BuiltinLintDiag::UnstableFeature(msg) => {
            lints::UnstableFeature { msg }.decorate_lint(diag);
        }
        BuiltinLintDiag::AvoidUsingIntelSyntax => {
            lints::AvoidIntelSyntax.decorate_lint(diag);
        }
        BuiltinLintDiag::AvoidUsingAttSyntax => {
            lints::AvoidAttSyntax.decorate_lint(diag);
        }
        BuiltinLintDiag::IncompleteInclude => {
            lints::IncompleteInclude.decorate_lint(diag);
        }
        BuiltinLintDiag::UnnameableTestItems => {
            lints::UnnameableTestItems.decorate_lint(diag);
        }
        BuiltinLintDiag::DuplicateMacroAttribute => {
            lints::DuplicateMacroAttribute.decorate_lint(diag);
        }
        BuiltinLintDiag::CfgAttrNoAttributes => {
            lints::CfgAttrNoAttributes.decorate_lint(diag);
        }
        BuiltinLintDiag::MetaVariableStillRepeating(name) => {
            lints::MetaVariableStillRepeating { name }.decorate_lint(diag);
        }
        BuiltinLintDiag::MetaVariableWrongOperator => {
            lints::MetaVariableWrongOperator.decorate_lint(diag);
        }
        BuiltinLintDiag::DuplicateMatcherBinding => {
            lints::DuplicateMatcherBinding.decorate_lint(diag);
        }
        BuiltinLintDiag::UnknownMacroVariable(name) => {
            lints::UnknownMacroVariable { name }.decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedCrateDependency { extern_crate, local_crate } => {
            lints::UnusedCrateDependency { extern_crate, local_crate }.decorate_lint(diag)
        }
        BuiltinLintDiag::IllFormedAttributeInput { suggestions } => {
            lints::IllFormedAttributeInput {
                num_suggestions: suggestions.len(),
                suggestions: DiagArgValue::StrListSepByAnd(
                    suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                ),
            }
            .decorate_lint(diag)
        }
        BuiltinLintDiag::InnerAttributeUnstable { is_macro } => if is_macro {
            lints::InnerAttributeUnstable::InnerMacroAttribute
        } else {
            lints::InnerAttributeUnstable::CustomInnerAttribute
        }
        .decorate_lint(diag),
        BuiltinLintDiag::OutOfScopeMacroCalls { span, path, location } => {
            lints::OutOfScopeMacroCalls { span, path, location }.decorate_lint(diag)
        }
        BuiltinLintDiag::UnexpectedBuiltinCfg { cfg, cfg_name, controlled_by } => {
            lints::UnexpectedBuiltinCfg { cfg, cfg_name, controlled_by }.decorate_lint(diag)
        }
    }
}
