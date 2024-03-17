#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{add_elided_lifetime_in_path_suggestion, Diag};
use rustc_errors::{Applicability, SuggestionStyle};
use rustc_middle::middle::stability;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::Session;
use rustc_span::BytePos;

mod check_cfg;

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
        BuiltinLintDiag::Normal => (),
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
        BuiltinLintDiag::ProcMacroDeriveResolutionFallback(span) => {
            diag.span_label(
                span,
                "names from parent modules are not accessible without an explicit import",
            );
        }
        BuiltinLintDiag::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def) => {
            diag.span_note(span_def, "the macro is defined here");
        }
        BuiltinLintDiag::ElidedLifetimesInPaths(n, path_span, incl_angl_brckt, insertion_span) => {
            add_elided_lifetime_in_path_suggestion(
                sess.source_map(),
                diag,
                n,
                path_span,
                incl_angl_brckt,
                insertion_span,
            );
        }
        BuiltinLintDiag::UnknownCrateTypes(span, note, sugg) => {
            diag.span_suggestion(span, note, sugg, Applicability::MaybeIncorrect);
        }
        BuiltinLintDiag::UnusedImports(message, replaces, in_test_module) => {
            if !replaces.is_empty() {
                diag.tool_only_multipart_suggestion(
                    message,
                    replaces,
                    Applicability::MachineApplicable,
                );
            }

            if let Some(span) = in_test_module {
                diag.span_help(
                    sess.source_map().guess_head_span(span),
                    "if this is a test module, consider adding a `#[cfg(test)]` to the containing module",
                );
            }
        }
        BuiltinLintDiag::RedundantImport(spans, ident) => {
            for (span, is_imported) in spans {
                let introduced = if is_imported { "imported" } else { "defined" };
                let span_msg = if span.is_dummy() { "by prelude" } else { "here" };
                diag.span_label(
                    span,
                    format!("the item `{ident}` is already {introduced} {span_msg}"),
                );
            }
        }
        BuiltinLintDiag::DeprecatedMacro(suggestion, span) => {
            stability::deprecation_suggestion(diag, "macro", suggestion, span)
        }
        BuiltinLintDiag::UnusedDocComment(span) => {
            diag.span_label(span, "rustdoc does not generate documentation for macro invocations");
            diag.help("to document an item produced by a macro, \
                                  the macro must produce the documentation as part of its expansion");
        }
        BuiltinLintDiag::PatternsInFnsWithoutBody(span, ident) => {
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
        BuiltinLintDiag::ReservedPrefix(span) => {
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
        BuiltinLintDiag::NamedAsmLabel(help) => {
            diag.help(help);
            diag.note("see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information");
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
        } => {
            debug!(?param_span, ?use_span, ?deletion_span);
            diag.span_label(param_span, "this lifetime...");
            diag.span_label(use_span, "...is used only here");
            if let Some(deletion_span) = deletion_span {
                let msg = "elide the single-use lifetime";
                let (use_span, replace_lt) = if elide {
                    let use_span = sess
                        .source_map()
                        .span_extend_while(use_span, char::is_whitespace)
                        .unwrap_or(use_span);
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
        BuiltinLintDiag::SingleUseLifetime { param_span: _, use_span: None, deletion_span } => {
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
        BuiltinLintDiag::ByteSliceInPackedStructWithDerive => {
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
        BuiltinLintDiag::RedundantImportVisibility { max_vis, span } => {
            diag.span_note(span, format!("the most public imported item is `{max_vis}`"));
            diag.help(
                "reduce the glob import's visibility or increase visibility of imported items",
            );
        }
    }
}
