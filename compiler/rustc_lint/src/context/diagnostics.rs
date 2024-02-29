#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_errors::{add_elided_lifetime_in_path_suggestion, Diag};
use rustc_errors::{Applicability, SuggestionStyle};
use rustc_middle::middle::stability;
use rustc_session::config::ExpectedValues;
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::Session;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::BytePos;

pub(super) fn builtin(sess: &Session, diagnostic: BuiltinLintDiagnostics, diag: &mut Diag<'_, ()>) {
    match diagnostic {
        BuiltinLintDiagnostics::UnicodeTextFlow(span, content) => {
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
        BuiltinLintDiagnostics::Normal => (),
        BuiltinLintDiagnostics::AbsPathWithModule(span) => {
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
        BuiltinLintDiagnostics::ProcMacroDeriveResolutionFallback(span) => {
            diag.span_label(
                span,
                "names from parent modules are not accessible without an explicit import",
            );
        }
        BuiltinLintDiagnostics::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def) => {
            diag.span_note(span_def, "the macro is defined here");
        }
        BuiltinLintDiagnostics::ElidedLifetimesInPaths(
            n,
            path_span,
            incl_angl_brckt,
            insertion_span,
        ) => {
            add_elided_lifetime_in_path_suggestion(
                sess.source_map(),
                diag,
                n,
                path_span,
                incl_angl_brckt,
                insertion_span,
            );
        }
        BuiltinLintDiagnostics::UnknownCrateTypes(span, note, sugg) => {
            diag.span_suggestion(span, note, sugg, Applicability::MaybeIncorrect);
        }
        BuiltinLintDiagnostics::UnusedImports(message, replaces, in_test_module) => {
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
                    "consider adding a `#[cfg(test)]` to the containing module",
                );
            }
        }
        BuiltinLintDiagnostics::RedundantImport(spans, ident) => {
            for (span, is_imported) in spans {
                let introduced = if is_imported { "imported" } else { "defined" };
                diag.span_label(span, format!("the item `{ident}` is already {introduced} here"));
            }
        }
        BuiltinLintDiagnostics::DeprecatedMacro(suggestion, span) => {
            stability::deprecation_suggestion(diag, "macro", suggestion, span)
        }
        BuiltinLintDiagnostics::UnusedDocComment(span) => {
            diag.span_label(span, "rustdoc does not generate documentation for macro invocations");
            diag.help("to document an item produced by a macro, \
                                  the macro must produce the documentation as part of its expansion");
        }
        BuiltinLintDiagnostics::PatternsInFnsWithoutBody(span, ident) => {
            diag.span_suggestion(
                span,
                "remove `mut` from the parameter",
                ident,
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::MissingAbi(span, default_abi) => {
            diag.span_label(span, "ABI should be specified here");
            diag.help(format!("the default ABI is {}", default_abi.name()));
        }
        BuiltinLintDiagnostics::LegacyDeriveHelpers(span) => {
            diag.span_label(span, "the attribute is introduced here");
        }
        BuiltinLintDiagnostics::ProcMacroBackCompat(note) => {
            diag.note(note);
        }
        BuiltinLintDiagnostics::OrPatternsBackCompat(span, suggestion) => {
            diag.span_suggestion(
                span,
                "use pat_param to preserve semantics",
                suggestion,
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::ReservedPrefix(span) => {
            diag.span_label(span, "unknown prefix");
            diag.span_suggestion_verbose(
                span.shrink_to_hi(),
                "insert whitespace here to avoid this being parsed as a prefix in Rust 2021",
                " ",
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::UnusedBuiltinAttribute { attr_name, macro_name, invoc_span } => {
            diag.span_note(
                        invoc_span,
                        format!("the built-in attribute `{attr_name}` will be ignored, since it's applied to the macro invocation `{macro_name}`")
                    );
        }
        BuiltinLintDiagnostics::TrailingMacro(is_trailing, name) => {
            if is_trailing {
                diag.note("macro invocations at the end of a block are treated as expressions");
                diag.note(format!("to ignore the value produced by the macro, add a semicolon after the invocation of `{name}`"));
            }
        }
        BuiltinLintDiagnostics::BreakWithLabelAndLoop(span) => {
            diag.multipart_suggestion(
                "wrap this expression in parentheses",
                vec![
                    (span.shrink_to_lo(), "(".to_string()),
                    (span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::NamedAsmLabel(help) => {
            diag.help(help);
            diag.note("see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information");
        }
        BuiltinLintDiagnostics::UnexpectedCfgName((name, name_span), value) => {
            #[allow(rustc::potential_query_instability)]
            let possibilities: Vec<Symbol> =
                sess.parse_sess.check_config.expecteds.keys().copied().collect();

            let mut names_possibilities: Vec<_> = if value.is_none() {
                // We later sort and display all the possibilities, so the order here does not matter.
                #[allow(rustc::potential_query_instability)]
                sess.parse_sess
                    .check_config
                    .expecteds
                    .iter()
                    .filter_map(|(k, v)| match v {
                        ExpectedValues::Some(v) if v.contains(&Some(name)) => Some(k),
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let is_from_cargo = rustc_session::utils::was_invoked_from_cargo();
            let mut is_feature_cfg = name == sym::feature;

            if is_feature_cfg && is_from_cargo {
                diag.help("consider defining some features in `Cargo.toml`");
            // Suggest the most probable if we found one
            } else if let Some(best_match) = find_best_match_for_name(&possibilities, name, None) {
                if let Some(ExpectedValues::Some(best_match_values)) =
                    sess.parse_sess.check_config.expecteds.get(&best_match)
                {
                    // We will soon sort, so the initial order does not matter.
                    #[allow(rustc::potential_query_instability)]
                    let mut possibilities =
                        best_match_values.iter().flatten().map(Symbol::as_str).collect::<Vec<_>>();
                    possibilities.sort();

                    let mut should_print_possibilities = true;
                    if let Some((value, value_span)) = value {
                        if best_match_values.contains(&Some(value)) {
                            diag.span_suggestion(
                                name_span,
                                "there is a config with a similar name and value",
                                best_match,
                                Applicability::MaybeIncorrect,
                            );
                            should_print_possibilities = false;
                        } else if best_match_values.contains(&None) {
                            diag.span_suggestion(
                                name_span.to(value_span),
                                "there is a config with a similar name and no value",
                                best_match,
                                Applicability::MaybeIncorrect,
                            );
                            should_print_possibilities = false;
                        } else if let Some(first_value) = possibilities.first() {
                            diag.span_suggestion(
                                name_span.to(value_span),
                                "there is a config with a similar name and different values",
                                format!("{best_match} = \"{first_value}\""),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            diag.span_suggestion(
                                name_span.to(value_span),
                                "there is a config with a similar name and different values",
                                best_match,
                                Applicability::MaybeIncorrect,
                            );
                        };
                    } else {
                        diag.span_suggestion(
                            name_span,
                            "there is a config with a similar name",
                            best_match,
                            Applicability::MaybeIncorrect,
                        );
                    }

                    if !possibilities.is_empty() && should_print_possibilities {
                        let possibilities = possibilities.join("`, `");
                        diag.help(format!(
                            "expected values for `{best_match}` are: `{possibilities}`"
                        ));
                    }
                } else {
                    diag.span_suggestion(
                        name_span,
                        "there is a config with a similar name",
                        best_match,
                        Applicability::MaybeIncorrect,
                    );
                }

                is_feature_cfg |= best_match == sym::feature;
            } else {
                if !names_possibilities.is_empty() && names_possibilities.len() <= 3 {
                    names_possibilities.sort();
                    for cfg_name in names_possibilities.iter() {
                        diag.span_suggestion(
                            name_span,
                            "found config with similar value",
                            format!("{cfg_name} = \"{name}\""),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                if !possibilities.is_empty() {
                    let mut possibilities =
                        possibilities.iter().map(Symbol::as_str).collect::<Vec<_>>();
                    possibilities.sort();
                    let possibilities = possibilities.join("`, `");

                    // The list of expected names can be long (even by default) and
                    // so the diagnostic produced can take a lot of space. To avoid
                    // cloging the user output we only want to print that diagnostic
                    // once.
                    diag.help_once(format!("expected names are: `{possibilities}`"));
                }
            }

            let inst = if let Some((value, _value_span)) = value {
                let pre = if is_from_cargo { "\\" } else { "" };
                format!("cfg({name}, values({pre}\"{value}{pre}\"))")
            } else {
                format!("cfg({name})")
            };

            if is_from_cargo {
                if !is_feature_cfg {
                    diag.help(format!("consider using a Cargo feature instead or adding `println!(\"cargo:rustc-check-cfg={inst}\");` to the top of a `build.rs`"));
                }
                diag.note("see <https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#check-cfg> for more information about checking conditional configuration");
            } else {
                diag.help(format!("to expect this configuration use `--check-cfg={inst}`"));
                diag.note("see <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/check-cfg.html> for more information about checking conditional configuration");
            }
        }
        BuiltinLintDiagnostics::UnexpectedCfgValue((name, name_span), value) => {
            let Some(ExpectedValues::Some(values)) =
                &sess.parse_sess.check_config.expecteds.get(&name)
            else {
                bug!(
                    "it shouldn't be possible to have a diagnostic on a value whose name is not in values"
                );
            };
            let mut have_none_possibility = false;
            // We later sort possibilities if it is not empty, so the
            // order here does not matter.
            #[allow(rustc::potential_query_instability)]
            let possibilities: Vec<Symbol> = values
                .iter()
                .inspect(|a| have_none_possibility |= a.is_none())
                .copied()
                .flatten()
                .collect();
            let is_from_cargo = rustc_session::utils::was_invoked_from_cargo();

            // Show the full list if all possible values for a given name, but don't do it
            // for names as the possibilities could be very long
            if !possibilities.is_empty() {
                {
                    let mut possibilities =
                        possibilities.iter().map(Symbol::as_str).collect::<Vec<_>>();
                    possibilities.sort();

                    let possibilities = possibilities.join("`, `");
                    let none = if have_none_possibility { "(none), " } else { "" };

                    diag.note(format!("expected values for `{name}` are: {none}`{possibilities}`"));
                }

                if let Some((value, value_span)) = value {
                    // Suggest the most probable if we found one
                    if let Some(best_match) = find_best_match_for_name(&possibilities, value, None)
                    {
                        diag.span_suggestion(
                            value_span,
                            "there is a expected value with a similar name",
                            format!("\"{best_match}\""),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else if let &[first_possibility] = &possibilities[..] {
                    diag.span_suggestion(
                        name_span.shrink_to_hi(),
                        "specify a config value",
                        format!(" = \"{first_possibility}\""),
                        Applicability::MaybeIncorrect,
                    );
                }
            } else if have_none_possibility {
                diag.note(format!("no expected value for `{name}`"));
                if let Some((_value, value_span)) = value {
                    diag.span_suggestion(
                        name_span.shrink_to_hi().to(value_span),
                        "remove the value",
                        "",
                        Applicability::MaybeIncorrect,
                    );
                }
            } else {
                diag.note(format!("no expected values for `{name}`"));

                let sp = if let Some((_value, value_span)) = value {
                    name_span.to(value_span)
                } else {
                    name_span
                };
                diag.span_suggestion(sp, "remove the condition", "", Applicability::MaybeIncorrect);
            }

            // We don't want to suggest adding values to well known names
            // since those are defined by rustc it-self. Users can still
            // do it if they want, but should not encourage them.
            let is_cfg_a_well_know_name =
                sess.parse_sess.check_config.well_known_names.contains(&name);

            let inst = if let Some((value, _value_span)) = value {
                let pre = if is_from_cargo { "\\" } else { "" };
                format!("cfg({name}, values({pre}\"{value}{pre}\"))")
            } else {
                format!("cfg({name})")
            };

            if is_from_cargo {
                if name == sym::feature {
                    if let Some((value, _value_span)) = value {
                        diag.help(format!(
                            "consider adding `{value}` as a feature in `Cargo.toml`"
                        ));
                    } else {
                        diag.help("consider defining some features in `Cargo.toml`");
                    }
                } else if !is_cfg_a_well_know_name {
                    diag.help(format!("consider using a Cargo feature instead or adding `println!(\"cargo:rustc-check-cfg={inst}\");` to the top of a `build.rs`"));
                }
                diag.note("see <https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#check-cfg> for more information about checking conditional configuration");
            } else {
                if !is_cfg_a_well_know_name {
                    diag.help(format!("to expect this configuration use `--check-cfg={inst}`"));
                }
                diag.note("see <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/check-cfg.html> for more information about checking conditional configuration");
            }
        }
        BuiltinLintDiagnostics::DeprecatedWhereclauseLocation(sugg) => {
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
        BuiltinLintDiagnostics::SingleUseLifetime {
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
        BuiltinLintDiagnostics::SingleUseLifetime {
            param_span: _,
            use_span: None,
            deletion_span,
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
        BuiltinLintDiagnostics::NamedArgumentUsedPositionally {
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
        BuiltinLintDiagnostics::ByteSliceInPackedStructWithDerive => {
            diag.help("consider implementing the trait by hand, or remove the `packed` attribute");
        }
        BuiltinLintDiagnostics::UnusedExternCrate { removal_span } => {
            diag.span_suggestion(removal_span, "remove it", "", Applicability::MachineApplicable);
        }
        BuiltinLintDiagnostics::ExternCrateNotIdiomatic { vis_span, ident_span } => {
            let suggestion_span = vis_span.between(ident_span);
            diag.span_suggestion_verbose(
                suggestion_span,
                "convert it to a `use`",
                if vis_span.is_empty() { "use " } else { " use " },
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::AmbiguousGlobImports { diag: ambiguity } => {
            rustc_errors::report_ambiguity_error(diag, ambiguity);
        }
        BuiltinLintDiagnostics::AmbiguousGlobReexports {
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
        BuiltinLintDiagnostics::HiddenGlobReexports {
            name,
            namespace,
            glob_reexport_span,
            private_item_span,
        } => {
            diag.span_note(glob_reexport_span, format!("the name `{name}` in the {namespace} namespace is supposed to be publicly re-exported here"));
            diag.span_note(private_item_span, "but the private item here shadows it".to_owned());
        }
        BuiltinLintDiagnostics::UnusedQualifications { removal_span } => {
            diag.span_suggestion_verbose(
                removal_span,
                "remove the unnecessary path segments",
                "",
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::AssociatedConstElidedLifetime { elided, span } => {
            diag.span_suggestion_verbose(
                if elided { span.shrink_to_hi() } else { span },
                "use the `'static` lifetime",
                if elided { "'static " } else { "'static" },
                Applicability::MachineApplicable,
            );
        }
        BuiltinLintDiagnostics::RedundantImportVisibility { max_vis, span } => {
            diag.span_note(span, format!("the most public imported item is `{max_vis}`"));
            diag.help(
                "reduce the glob import's visibility or increase visibility of imported items",
            );
        }
    }
}
