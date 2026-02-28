use std::borrow::Cow;

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{Applicability, DiagArgValue, elided_lifetime_in_path_suggestion};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
use rustc_middle::middle::stability;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_span::BytePos;
use tracing::debug;

use crate::{EmitDiag, lints};

mod check_cfg;

pub fn decorate_builtin_lint(
    ctx: impl EmitDiag,
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    diagnostic: BuiltinLintDiag,
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

            ctx.emit(lints::UnicodeTextFlow {
                comment_span,
                characters,
                suggestions,
                num_codepoints: spans.len(),
            });
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
            ctx.emit(lints::AbsPathWithModule {
                sugg: lints::AbsPathWithModuleSugg { span: mod_span, applicability, replacement },
            });
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(n, path_span, incl_angl_brckt, insertion_span) => {
            ctx.emit(lints::ElidedLifetimesInPaths {
                subdiag: elided_lifetime_in_path_suggestion(
                    sess.source_map(),
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ),
            });
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

            ctx.emit(lints::UnusedImports {
                sugg,
                test_module_span,
                num_snippets: span_snippets.len(),
                span_snippets: DiagArgValue::StrListSepByAnd(
                    span_snippets.into_iter().map(Cow::Owned).collect(),
                ),
            });
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
            ctx.emit(lints::RedundantImport { subs, ident });
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

            ctx.emit(stability::Deprecated {
                sub,
                kind: "macro".to_owned(),
                path,
                note,
                since_kind,
            });
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody { span: remove_span, ident, is_foreign } => {
            let sub = lints::PatternsInFnsWithoutBodySub { ident, span: remove_span };
            ctx.emit(if is_foreign {
                lints::PatternsInFnsWithoutBody::Foreign { sub }
            } else {
                lints::PatternsInFnsWithoutBody::Bodiless { sub }
            });
        }
        BuiltinLintDiag::ReservedPrefix(label_span, prefix) => {
            ctx.emit(lints::ReservedPrefix {
                label: label_span,
                suggestion: label_span.shrink_to_hi(),
                prefix,
            });
        }
        BuiltinLintDiag::RawPrefix(label_span) => {
            ctx.emit(lints::RawPrefix { label: label_span, suggestion: label_span.shrink_to_hi() });
        }
        BuiltinLintDiag::ReservedString { is_string, suggestion } => {
            if is_string {
                ctx.emit(lints::ReservedString { suggestion });
            } else {
                ctx.emit(lints::ReservedMultihash { suggestion });
            }
        }
        BuiltinLintDiag::BreakWithLabelAndLoop(sugg_span) => {
            ctx.emit(lints::BreakWithLabelAndLoop {
                sub: lints::BreakWithLabelAndLoopSub {
                    left: sugg_span.shrink_to_lo(),
                    right: sugg_span.shrink_to_hi(),
                },
            });
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
            ctx.emit(lints::DeprecatedWhereClauseLocation { suggestion });
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

            ctx.emit(lints::SingleUseLifetime { suggestion, param_span, use_span, ident });
        }
        BuiltinLintDiag::SingleUseLifetime { use_span: None, deletion_span, ident, .. } => {
            ctx.emit(lints::UnusedLifetime { deletion_span, ident });
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

            ctx.emit(lints::NamedArgumentUsedPositionally {
                named_arg_sp,
                position_label_sp: position_sp_for_msg,
                suggestion,
                name,
                named_arg_name,
            });
        }
        BuiltinLintDiag::AmbiguousGlobReexports {
            name,
            namespace,
            first_reexport_span,
            duplicate_reexport_span,
        } => {
            ctx.emit(lints::AmbiguousGlobReexports {
                first_reexport: first_reexport_span,
                duplicate_reexport: duplicate_reexport_span,
                name,
                namespace,
            });
        }
        BuiltinLintDiag::HiddenGlobReexports {
            name,
            namespace,
            glob_reexport_span,
            private_item_span,
        } => {
            ctx.emit(lints::HiddenGlobReexports {
                glob_reexport: glob_reexport_span,
                private_item: private_item_span,

                name,
                namespace,
            });
        }
        BuiltinLintDiag::UnusedQualifications { removal_span } => {
            ctx.emit(lints::UnusedQualifications { removal_span });
        }
        BuiltinLintDiag::AssociatedConstElidedLifetime {
            elided,
            span: lt_span,
            lifetimes_in_scope,
        } => {
            let lt_span = if elided { lt_span.shrink_to_hi() } else { lt_span };
            let code = if elided { "'static " } else { "'static" };
            ctx.emit(lints::AssociatedConstElidedLifetime {
                span: lt_span,
                code,
                elided,
                lifetimes_in_scope,
            });
        }
        BuiltinLintDiag::UnreachableCfg { span, wildcard_span } => match wildcard_span {
            Some(wildcard_span) => {
                ctx.emit(lints::UnreachableCfgSelectPredicateWildcard { span, wildcard_span })
            }
            None => ctx.emit(lints::UnreachableCfgSelectPredicate { span }),
        },

        BuiltinLintDiag::UnusedCrateDependency { extern_crate, local_crate } => {
            ctx.emit(lints::UnusedCrateDependency { extern_crate, local_crate })
        }
        BuiltinLintDiag::UnusedVisibility(span) => ctx.emit(lints::UnusedVisibility { span }),
        BuiltinLintDiag::AttributeLint(kind) => decorate_attribute_lint(ctx, sess, tcx, &kind),
    }
}

pub fn decorate_attribute_lint(
    ctx: impl EmitDiag,
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    kind: &AttributeLintKind,
) {
    match kind {
        &AttributeLintKind::UnusedDuplicate { this, other, warning } => {
            ctx.emit(lints::UnusedDuplicate { this, other, warning })
        }
        AttributeLintKind::IllFormedAttributeInput { suggestions, docs } => {
            ctx.emit(lints::IllFormedAttributeInput {
                num_suggestions: suggestions.len(),
                suggestions: DiagArgValue::StrListSepByAnd(
                    suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                ),
                has_docs: docs.is_some(),
                docs: docs.unwrap_or(""),
            })
        }
        AttributeLintKind::EmptyAttribute { first_span, attr_path, valid_without_list } => ctx
            .emit(lints::EmptyAttributeList {
                attr_span: *first_span,
                attr_path: attr_path.clone(),
                valid_without_list: *valid_without_list,
            }),
        AttributeLintKind::InvalidTarget { name, target, applied, only, attr_span } => {
            ctx.emit(lints::InvalidTargetLint {
                name: name.clone(),
                target,
                applied: DiagArgValue::StrListSepByAnd(
                    applied.into_iter().map(|i| Cow::Owned(i.to_string())).collect(),
                ),
                only,
                attr_span: *attr_span,
            })
        }
        &AttributeLintKind::InvalidStyle { ref name, is_used_as_inner, target, target_span } => ctx
            .emit(lints::InvalidAttrStyle {
                name: name.clone(),
                is_used_as_inner,
                target_span: (!is_used_as_inner).then_some(target_span),
                target,
            }),
        &AttributeLintKind::UnsafeAttrOutsideUnsafe { attribute_name_span, sugg_spans } => ctx
            .emit(lints::UnsafeAttrOutsideUnsafeLint {
                span: attribute_name_span,
                suggestion: sugg_spans
                    .map(|(left, right)| lints::UnsafeAttrOutsideUnsafeSuggestion { left, right }),
            }),
        &AttributeLintKind::UnexpectedCfgName(name, value) => {
            ctx.emit(check_cfg::unexpected_cfg_name(sess, tcx, name, value))
        }
        &AttributeLintKind::UnexpectedCfgValue(name, value) => {
            ctx.emit(check_cfg::unexpected_cfg_value(sess, tcx, name, value))
        }
        &AttributeLintKind::DuplicateDocAlias { first_definition } => {
            ctx.emit(lints::DocAliasDuplicated { first_defn: first_definition })
        }

        &AttributeLintKind::DocAutoCfgExpectsHideOrShow => {
            ctx.emit(lints::DocAutoCfgExpectsHideOrShow)
        }

        &AttributeLintKind::AmbiguousDeriveHelpers => ctx.emit(lints::AmbiguousDeriveHelpers),

        &AttributeLintKind::DocAutoCfgHideShowUnexpectedItem { attr_name } => {
            ctx.emit(lints::DocAutoCfgHideShowUnexpectedItem { attr_name })
        }

        &AttributeLintKind::DocAutoCfgHideShowExpectsList { attr_name } => {
            ctx.emit(lints::DocAutoCfgHideShowExpectsList { attr_name })
        }

        &AttributeLintKind::DocInvalid => ctx.emit(lints::DocInvalid),

        &AttributeLintKind::DocUnknownInclude { span, inner, value } => {
            ctx.emit(lints::DocUnknownInclude {
                inner,
                value,
                sugg: (span, Applicability::MaybeIncorrect),
            })
        }

        &AttributeLintKind::DocUnknownSpotlight { span } => {
            ctx.emit(lints::DocUnknownSpotlight { sugg_span: span })
        }

        &AttributeLintKind::DocUnknownPasses { name, span } => {
            ctx.emit(lints::DocUnknownPasses { name, note_span: span })
        }

        &AttributeLintKind::DocUnknownPlugins { span } => {
            ctx.emit(lints::DocUnknownPlugins { label_span: span })
        }

        &AttributeLintKind::DocUnknownAny { name } => ctx.emit(lints::DocUnknownAny { name }),

        &AttributeLintKind::DocAutoCfgWrongLiteral => ctx.emit(lints::DocAutoCfgWrongLiteral),

        &AttributeLintKind::DocTestTakesList => ctx.emit(lints::DocTestTakesList),

        &AttributeLintKind::DocTestUnknown { name } => ctx.emit(lints::DocTestUnknown { name }),

        &AttributeLintKind::DocTestLiteral => ctx.emit(lints::DocTestLiteral),

        &AttributeLintKind::AttrCrateLevelOnly => ctx.emit(lints::AttrCrateLevelOnly),

        &AttributeLintKind::DoNotRecommendDoesNotExpectArgs => {
            ctx.emit(lints::DoNotRecommendDoesNotExpectArgs)
        }

        &AttributeLintKind::CrateTypeUnknown { span, suggested } => {
            ctx.emit(lints::UnknownCrateTypes {
                sugg: suggested.map(|s| lints::UnknownCrateTypesSuggestion { span, snippet: s }),
            })
        }

        &AttributeLintKind::MalformedDoc => ctx.emit(lints::MalformedDoc),

        &AttributeLintKind::ExpectedNoArgs => ctx.emit(lints::ExpectedNoArgs),

        &AttributeLintKind::ExpectedNameValue => ctx.emit(lints::ExpectedNameValue),
        &AttributeLintKind::MalformedOnUnimplementedAttr { span } => {
            ctx.emit(lints::MalformedOnUnimplementedAttrLint { span })
        }
        &AttributeLintKind::MalformedOnConstAttr { span } => {
            ctx.emit(lints::MalformedOnConstAttrLint { span })
        }
        AttributeLintKind::MalformedDiagnosticFormat { warning } => match warning {
            FormatWarning::PositionalArgument { .. } => {
                ctx.emit(lints::DisallowedPositionalArgument)
            }
            FormatWarning::InvalidSpecifier { .. } => ctx.emit(lints::InvalidFormatSpecifier),
        },
        AttributeLintKind::DiagnosticWrappedParserError { description, label, span } => {
            ctx.emit(lints::WrappedParserError { description, label, span: *span })
        }
        &AttributeLintKind::IgnoredDiagnosticOption { option_name, first_span, later_span } => {
            ctx.emit(lints::IgnoredDiagnosticOption { option_name, first_span, later_span })
        }
        &AttributeLintKind::MissingOptionsForOnUnimplemented => {
            ctx.emit(lints::MissingOptionsForOnUnimplementedAttr)
        }
        &AttributeLintKind::MissingOptionsForOnConst => {
            ctx.emit(lints::MissingOptionsForOnConstAttr)
        }
    }
}
