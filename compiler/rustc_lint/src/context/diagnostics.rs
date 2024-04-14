#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{
    elided_lifetime_in_path_suggestion, pluralize, Diag, DiagMessage, LintDiagnostic,
};
use rustc_errors::{Applicability, SuggestionStyle};
use rustc_middle::middle::stability;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::Session;
use rustc_span::BytePos;

use std::fmt::Write;

mod check_cfg;

#[cfg(test)]
mod tests;

pub(super) fn builtin(sess: &Session, diagnostic: BuiltinLintDiag, diag: &mut Diag<'_, ()>) {
    match diagnostic {
        BuiltinLintDiag::UnicodeTextFlow(span, content) => {
            let spans: Vec<_> = content
                .char_indices()
                .filter_map(|(i, c)| {
                    TEXT_FLOW_CONTROL_CHARS.contains(&c).then(|| {
                        let lo = span.lo() + BytePos(2 + i as u32);
                        (c, span.with_lo(lo).with_hi(lo + BytePos(c.len_utf8() as u32)))
                    })
                })
                .collect();
            let (an, s) = match spans.len() {
                1 => ("an ", ""),
                _ => ("", "s"),
            };
            diag.span_label(
                span,
                format!(
                    "this comment contains {an}invisible unicode text flow control codepoint{s}",
                ),
            );
            for (c, span) in &spans {
                diag.span_label(*span, format!("{c:?}"));
            }
            diag.note(
                "these kind of unicode codepoints change the way text flows on \
                         applications that support them, but can cause confusion because they \
                         change the order of characters on the screen",
            );
            if !spans.is_empty() {
                diag.multipart_suggestion_with_style(
                    "if their presence wasn't intentional, you can remove them",
                    spans.into_iter().map(|(_, span)| (span, "".to_string())).collect(),
                    Applicability::MachineApplicable,
                    SuggestionStyle::HideCodeAlways,
                );
            }
        }
        BuiltinLintDiag::AbsPathWithModule(span) => {
            let (sugg, app) = match sess.source_map().span_to_snippet(span) {
                Ok(ref s) => {
                    // FIXME(Manishearth) ideally the emitting code
                    // can tell us whether or not this is global
                    let opt_colon = if s.trim_start().starts_with("::") { "" } else { "::" };

                    (format!("crate{opt_colon}{s}"), Applicability::MachineApplicable)
                }
                Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders),
            };
            diag.span_suggestion(span, "use `crate`", sugg, app);
        }
        BuiltinLintDiag::ProcMacroDeriveResolutionFallback { span, .. } => {
            diag.span_label(
                span,
                "names from parent modules are not accessible without an explicit import",
            );
        }
        BuiltinLintDiag::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def) => {
            diag.span_note(span_def, "the macro is defined here");
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(n, path_span, incl_angl_brckt, insertion_span) => {
            diag.subdiagnostic(
                sess.dcx(),
                elided_lifetime_in_path_suggestion(
                    sess.source_map(),
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ),
            );
        }
        BuiltinLintDiag::UnknownCrateTypes { span, candidate } => {
            if let Some(candidate) = candidate {
                diag.span_suggestion(
                    span,
                    "did you mean",
                    format!(r#""{candidate}""#),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        BuiltinLintDiag::UnusedImports { fix_msg, fixes, test_module_span, .. } => {
            if !fixes.is_empty() {
                diag.tool_only_multipart_suggestion(
                    fix_msg,
                    fixes,
                    Applicability::MachineApplicable,
                );
            }

            if let Some(span) = test_module_span {
                diag.span_help(
                    sess.source_map().guess_head_span(span),
                    "if this is a test module, consider adding a `#[cfg(test)]` to the containing module",
                );
            }
        }
        BuiltinLintDiag::RedundantImport(spans, ident) => {
            for (span, is_imported) in spans {
                let introduced = if is_imported { "imported" } else { "defined" };
                let span_msg = if span.is_dummy() { "by the extern prelude" } else { "here" };
                diag.span_label(
                    span,
                    format!("the item `{ident}` is already {introduced} {span_msg}"),
                );
            }
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
            let deprecated =
                stability::Deprecated { sub, kind: "macro".to_owned(), path, note, since_kind };
            deprecated.decorate_lint(diag);
        }
        BuiltinLintDiag::UnusedDocComment(span) => {
            diag.span_label(span, "rustdoc does not generate documentation for macro invocations");
            diag.help("to document an item produced by a macro, \
                                  the macro must produce the documentation as part of its expansion");
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody { span, ident, .. } => {
            diag.span_suggestion(
                span,
                "remove `mut` from the parameter",
                ident,
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::MissingAbi(span, default_abi) => {
            diag.span_label(span, "ABI should be specified here");
            diag.help(format!("the default ABI is {}", default_abi.name()));
        }
        BuiltinLintDiag::LegacyDeriveHelpers(span) => {
            diag.span_label(span, "the attribute is introduced here");
        }
        BuiltinLintDiag::ProcMacroBackCompat(note) => {
            diag.note(note);
        }
        BuiltinLintDiag::OrPatternsBackCompat(span, suggestion) => {
            diag.span_suggestion(
                span,
                "use pat_param to preserve semantics",
                suggestion,
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::ReservedPrefix(span, _) => {
            diag.span_label(span, "unknown prefix");
            diag.span_suggestion_verbose(
                span.shrink_to_hi(),
                "insert whitespace here to avoid this being parsed as a prefix in Rust 2021",
                " ",
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::UnusedBuiltinAttribute { attr_name, macro_name, invoc_span } => {
            diag.span_note(
                        invoc_span,
                        format!("the built-in attribute `{attr_name}` will be ignored, since it's applied to the macro invocation `{macro_name}`")
                    );
        }
        BuiltinLintDiag::TrailingMacro(is_trailing, name) => {
            if is_trailing {
                diag.note("macro invocations at the end of a block are treated as expressions");
                diag.note(format!("to ignore the value produced by the macro, add a semicolon after the invocation of `{name}`"));
            }
        }
        BuiltinLintDiag::BreakWithLabelAndLoop(span) => {
            diag.multipart_suggestion(
                "wrap this expression in parentheses",
                vec![
                    (span.shrink_to_lo(), "(".to_string()),
                    (span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::UnexpectedCfgName(name, value) => {
            check_cfg::unexpected_cfg_name(sess, diag, name, value)
        }
        BuiltinLintDiag::UnexpectedCfgValue(name, value) => {
            check_cfg::unexpected_cfg_value(sess, diag, name, value)
        }
        BuiltinLintDiag::DeprecatedWhereclauseLocation(sugg) => {
            let left_sp = diag.span.primary_span().unwrap();
            match sugg {
                Some((right_sp, sugg)) => diag.multipart_suggestion(
                    "move it to the end of the type declaration",
                    vec![(left_sp, String::new()), (right_sp, sugg)],
                    Applicability::MachineApplicable,
                ),
                None => diag.span_suggestion(
                    left_sp,
                    "remove this `where`",
                    "",
                    Applicability::MachineApplicable,
                ),
            };
            diag.note("see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information");
        }
        BuiltinLintDiag::SingleUseLifetime {
            param_span,
            use_span: Some((use_span, elide)),
            deletion_span,
            ..
        } => {
            debug!(?param_span, ?use_span, ?deletion_span);
            diag.span_label(param_span, "this lifetime...");
            diag.span_label(use_span, "...is used only here");
            if let Some(deletion_span) = deletion_span {
                let msg = "elide the single-use lifetime";
                let (use_span, replace_lt) = if elide {
                    let use_span = sess.source_map().span_extend_while_whitespace(use_span);
                    (use_span, String::new())
                } else {
                    (use_span, "'_".to_owned())
                };
                debug!(?deletion_span, ?use_span);

                // issue 107998 for the case such as a wrong function pointer type
                // `deletion_span` is empty and there is no need to report lifetime uses here
                let suggestions = if deletion_span.is_empty() {
                    vec![(use_span, replace_lt)]
                } else {
                    vec![(deletion_span, String::new()), (use_span, replace_lt)]
                };
                diag.multipart_suggestion(msg, suggestions, Applicability::MachineApplicable);
            }
        }
        BuiltinLintDiag::SingleUseLifetime {
            param_span: _, use_span: None, deletion_span, ..
        } => {
            debug!(?deletion_span);
            if let Some(deletion_span) = deletion_span {
                diag.span_suggestion(
                    deletion_span,
                    "elide the unused lifetime",
                    "",
                    Applicability::MachineApplicable,
                );
            }
        }
        BuiltinLintDiag::NamedArgumentUsedPositionally {
            position_sp_to_replace,
            position_sp_for_msg,
            named_arg_sp,
            named_arg_name,
            is_formatting_arg,
        } => {
            diag.span_label(
                named_arg_sp,
                "this named argument is referred to by position in formatting string",
            );
            if let Some(positional_arg_for_msg) = position_sp_for_msg {
                let msg = format!(
                    "this formatting argument uses named argument `{named_arg_name}` by position"
                );
                diag.span_label(positional_arg_for_msg, msg);
            }

            if let Some(positional_arg_to_replace) = position_sp_to_replace {
                let name = if is_formatting_arg { named_arg_name + "$" } else { named_arg_name };
                let span_to_replace = if let Ok(positional_arg_content) =
                    sess.source_map().span_to_snippet(positional_arg_to_replace)
                    && positional_arg_content.starts_with(':')
                {
                    positional_arg_to_replace.shrink_to_lo()
                } else {
                    positional_arg_to_replace
                };
                diag.span_suggestion_verbose(
                    span_to_replace,
                    "use the named argument by name to avoid ambiguity",
                    name,
                    Applicability::MaybeIncorrect,
                );
            }
        }
        BuiltinLintDiag::ByteSliceInPackedStructWithDerive { .. } => {
            diag.help("consider implementing the trait by hand, or remove the `packed` attribute");
        }
        BuiltinLintDiag::UnusedExternCrate { removal_span } => {
            diag.span_suggestion(removal_span, "remove it", "", Applicability::MachineApplicable);
        }
        BuiltinLintDiag::ExternCrateNotIdiomatic { vis_span, ident_span } => {
            let suggestion_span = vis_span.between(ident_span);
            diag.span_suggestion_verbose(
                suggestion_span,
                "convert it to a `use`",
                if vis_span.is_empty() { "use " } else { " use " },
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::AmbiguousGlobImports { diag: ambiguity } => {
            rustc_errors::report_ambiguity_error(diag, ambiguity);
        }
        BuiltinLintDiag::AmbiguousGlobReexports {
            name,
            namespace,
            first_reexport_span,
            duplicate_reexport_span,
        } => {
            diag.span_label(
                first_reexport_span,
                format!("the name `{name}` in the {namespace} namespace is first re-exported here"),
            );
            diag.span_label(
                duplicate_reexport_span,
                format!(
                    "but the name `{name}` in the {namespace} namespace is also re-exported here"
                ),
            );
        }
        BuiltinLintDiag::HiddenGlobReexports {
            name,
            namespace,
            glob_reexport_span,
            private_item_span,
        } => {
            diag.span_note(glob_reexport_span, format!("the name `{name}` in the {namespace} namespace is supposed to be publicly re-exported here"));
            diag.span_note(private_item_span, "but the private item here shadows it".to_owned());
        }
        BuiltinLintDiag::UnusedQualifications { removal_span } => {
            diag.span_suggestion_verbose(
                removal_span,
                "remove the unnecessary path segments",
                "",
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::AssociatedConstElidedLifetime { elided, span } => {
            diag.span_suggestion_verbose(
                if elided { span.shrink_to_hi() } else { span },
                "use the `'static` lifetime",
                if elided { "'static " } else { "'static" },
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiag::RedundantImportVisibility { max_vis, span, .. } => {
            diag.span_note(span, format!("the most public imported item is `{max_vis}`"));
            diag.help(
                "reduce the glob import's visibility or increase visibility of imported items",
            );
        }
        BuiltinLintDiag::UnknownDiagnosticAttribute { span, typo_name } => {
            if let Some(typo_name) = typo_name {
                diag.span_suggestion_verbose(
                    span,
                    "an attribute with a similar name exists",
                    typo_name,
                    Applicability::MachineApplicable,
                );
            }
        }
        BuiltinLintDiag::MacroUseDeprecated
        | BuiltinLintDiag::UnusedMacroUse
        | BuiltinLintDiag::PrivateExternCrateReexport(_)
        | BuiltinLintDiag::UnusedLabel
        | BuiltinLintDiag::MacroIsPrivate(_)
        | BuiltinLintDiag::UnusedMacroDefinition(_)
        | BuiltinLintDiag::MacroRuleNeverUsed(_, _)
        | BuiltinLintDiag::UnstableFeature(_)
        | BuiltinLintDiag::AvoidUsingIntelSyntax
        | BuiltinLintDiag::AvoidUsingAttSyntax
        | BuiltinLintDiag::IncompleteInclude
        | BuiltinLintDiag::UnnameableTestItems
        | BuiltinLintDiag::DuplicateMacroAttribute
        | BuiltinLintDiag::CfgAttrNoAttributes
        | BuiltinLintDiag::CrateTypeInCfgAttr
        | BuiltinLintDiag::CrateNameInCfgAttr
        | BuiltinLintDiag::MissingFragmentSpecifier
        | BuiltinLintDiag::MetaVariableStillRepeating(_)
        | BuiltinLintDiag::MetaVariableWrongOperator
        | BuiltinLintDiag::DuplicateMatcherBinding
        | BuiltinLintDiag::UnknownMacroVariable(_)
        | BuiltinLintDiag::UnusedExternCrate2 { .. }
        | BuiltinLintDiag::WasmCAbi
        | BuiltinLintDiag::IllFormedAttributeInput { .. }
        | BuiltinLintDiag::InnerAttributeUnstable { .. } => {}
    }
}

pub(super) fn builtin_message(diagnostic: &BuiltinLintDiag) -> DiagMessage {
    match diagnostic {
        BuiltinLintDiag::AbsPathWithModule(_) => {
            "absolute paths must start with `self`, `super`, `crate`, or an \
                external crate name in the 2018 edition"
                .into()
        }
        BuiltinLintDiag::ProcMacroDeriveResolutionFallback { ns, ident, .. } => {
            format!("cannot find {} `{}` in this scope", ns.descr(), ident).into()
        }
        BuiltinLintDiag::MacroExpandedMacroExportsAccessedByAbsolutePaths(_) => {
            "macro-expanded `macro_export` macros from the current crate cannot \
                be referred to by absolute paths"
                .into()
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(_, _, _, _) => {
            "hidden lifetime parameters in types are deprecated".into()
        }
        BuiltinLintDiag::UnknownCrateTypes { .. } => "invalid `crate_type` value".into(),
        BuiltinLintDiag::UnusedImports { span_snippets, .. } => format!(
            "unused import{}{}",
            pluralize!(span_snippets.len()),
            if !span_snippets.is_empty() {
                format!(": {}", span_snippets.join(", "))
            } else {
                String::new()
            }
        )
        .into(),
        BuiltinLintDiag::RedundantImport(_, source) => {
            format!("the item `{source}` is imported redundantly").into()
        }
        BuiltinLintDiag::DeprecatedMacro { since_kind, .. } => {
            stability::Deprecated::msg_for_since_kind(since_kind)
        }
        BuiltinLintDiag::MissingAbi(_, _) => crate::fluent_generated::lint_extern_without_abi,
        BuiltinLintDiag::UnusedDocComment(_) => "unused doc comment".into(),
        BuiltinLintDiag::UnusedBuiltinAttribute { attr_name, .. } => {
            format!("unused attribute `{attr_name}`").into()
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody { is_foreign, .. } => {
            if *is_foreign {
                crate::fluent_generated::lint_pattern_in_foreign
            } else {
                crate::fluent_generated::lint_pattern_in_bodiless
            }
        }
        BuiltinLintDiag::LegacyDeriveHelpers(_) => {
            "derive helper attribute is used before it is introduced".into()
        }
        BuiltinLintDiag::ProcMacroBackCompat(_) => "using an old version of `rental`".into(),
        BuiltinLintDiag::OrPatternsBackCompat(_, _) => {
            "the meaning of the `pat` fragment specifier is changing in Rust 2021, \
            which may affect this macro"
                .into()
        }
        BuiltinLintDiag::ReservedPrefix(_, prefix) => {
            format!("prefix `{prefix}` is unknown").into()
        }
        BuiltinLintDiag::TrailingMacro(_, _) => {
            "trailing semicolon in macro used in expression position".into()
        }
        BuiltinLintDiag::BreakWithLabelAndLoop(_) => {
            "this labeled break expression is easy to confuse with an unlabeled break with a \
            labeled value expression"
                .into()
        }
        BuiltinLintDiag::UnicodeTextFlow(_, _) => {
            "unicode codepoint changing visible direction of text present in comment".into()
        }
        BuiltinLintDiag::UnexpectedCfgName((name, _), _) => {
            format!("unexpected `cfg` condition name: `{}`", name).into()
        }
        BuiltinLintDiag::UnexpectedCfgValue(_, v) => if let Some((value, _)) = v {
            format!("unexpected `cfg` condition value: `{value}`")
        } else {
            format!("unexpected `cfg` condition value: (none)")
        }
        .into(),
        BuiltinLintDiag::DeprecatedWhereclauseLocation(_) => {
            crate::fluent_generated::lint_deprecated_where_clause_location
        }
        BuiltinLintDiag::SingleUseLifetime { use_span, ident, .. } => {
            if use_span.is_some() {
                format!("lifetime parameter `{}` only used once", ident).into()
            } else {
                format!("lifetime parameter `{}` never used", ident).into()
            }
        }
        BuiltinLintDiag::NamedArgumentUsedPositionally { named_arg_name, .. } => {
            format!("named argument `{}` is not used by name", named_arg_name).into()
        }
        BuiltinLintDiag::ByteSliceInPackedStructWithDerive { ty } => {
            format!("{ty} slice in a packed struct that derives a built-in trait").into()
        }
        BuiltinLintDiag::UnusedExternCrate { .. } => "unused extern crate".into(),
        BuiltinLintDiag::ExternCrateNotIdiomatic { .. } => {
            "`extern crate` is not idiomatic in the new edition".into()
        }
        BuiltinLintDiag::AmbiguousGlobImports { diag } => diag.msg.clone().into(),
        BuiltinLintDiag::AmbiguousGlobReexports { .. } => "ambiguous glob re-exports".into(),
        BuiltinLintDiag::HiddenGlobReexports { .. } => {
            "private item shadows public glob re-export".into()
        }
        BuiltinLintDiag::UnusedQualifications { .. } => "unnecessary qualification".into(),
        BuiltinLintDiag::AssociatedConstElidedLifetime { elided, .. } => if *elided {
            "`&` without an explicit lifetime name cannot be used here"
        } else {
            "`'_` cannot be used here"
        }
        .into(),
        BuiltinLintDiag::RedundantImportVisibility { import_vis, .. } => format!(
            "glob import doesn't reexport anything with visibility `{}` \
            because no imported item is public enough",
            import_vis
        )
        .into(),
        BuiltinLintDiag::MacroUseDeprecated => "deprecated `#[macro_use]` attribute used to \
                                import macros should be replaced at use sites \
                                with a `use` item to import the macro \
                                instead"
            .into(),
        BuiltinLintDiag::UnusedMacroUse => "unused `#[macro_use]` import".into(),
        BuiltinLintDiag::PrivateExternCrateReexport(ident) => format!(
            "extern crate `{ident}` is private, and cannot be \
                                   re-exported (error E0365), consider declaring with \
                                   `pub`"
        )
        .into(),
        BuiltinLintDiag::UnusedLabel => "unused label".into(),
        BuiltinLintDiag::MacroIsPrivate(ident) => format!("macro `{ident}` is private").into(),
        BuiltinLintDiag::UnusedMacroDefinition(name) => {
            format!("unused macro definition: `{}`", name).into()
        }
        BuiltinLintDiag::MacroRuleNeverUsed(n, name) => {
            format!("rule #{} of macro `{}` is never used", n + 1, name).into()
        }
        BuiltinLintDiag::UnstableFeature(msg) => msg.clone().into(),
        BuiltinLintDiag::AvoidUsingIntelSyntax => {
            "avoid using `.intel_syntax`, Intel syntax is the default".into()
        }
        BuiltinLintDiag::AvoidUsingAttSyntax => {
            "avoid using `.att_syntax`, prefer using `options(att_syntax)` instead".into()
        }
        BuiltinLintDiag::IncompleteInclude => {
            "include macro expected single expression in source".into()
        }
        BuiltinLintDiag::UnnameableTestItems => crate::fluent_generated::lint_unnameable_test_items,
        BuiltinLintDiag::DuplicateMacroAttribute => "duplicated attribute".into(),
        BuiltinLintDiag::CfgAttrNoAttributes => {
            crate::fluent_generated::lint_cfg_attr_no_attributes
        }
        BuiltinLintDiag::CrateTypeInCfgAttr => {
            crate::fluent_generated::lint_crate_type_in_cfg_attr_deprecated
        }
        BuiltinLintDiag::CrateNameInCfgAttr => {
            crate::fluent_generated::lint_crate_name_in_cfg_attr_deprecated
        }
        BuiltinLintDiag::MissingFragmentSpecifier => "missing fragment specifier".into(),
        BuiltinLintDiag::MetaVariableStillRepeating(name) => {
            format!("variable '{name}' is still repeating at this depth").into()
        }
        BuiltinLintDiag::MetaVariableWrongOperator => {
            "meta-variable repeats with different Kleene operator".into()
        }
        BuiltinLintDiag::DuplicateMatcherBinding => "duplicate matcher binding".into(),
        BuiltinLintDiag::UnknownMacroVariable(name) => {
            format!("unknown macro variable `{name}`").into()
        }
        BuiltinLintDiag::UnusedExternCrate2 { extern_crate, local_crate } => format!(
            "external crate `{}` unused in `{}`: remove the dependency or add `use {} as _;`",
            extern_crate, local_crate, extern_crate
        )
        .into(),
        BuiltinLintDiag::WasmCAbi => crate::fluent_generated::lint_wasm_c_abi,
        BuiltinLintDiag::IllFormedAttributeInput { suggestions } => suggestions
            .iter()
            .enumerate()
            .fold("attribute must be of the form ".to_string(), |mut acc, (i, sugg)| {
                if i != 0 {
                    write!(acc, " or ").unwrap();
                }
                write!(acc, "`{sugg}`").unwrap();
                acc
            })
            .into(),
        BuiltinLintDiag::InnerAttributeUnstable { is_macro } => if *is_macro {
            "inner macro attributes are unstable"
        } else {
            "custom inner attributes are unstable"
        }
        .into(),
        BuiltinLintDiag::UnknownDiagnosticAttribute { .. } => "unknown diagnostic attribute".into(),
    }
}
