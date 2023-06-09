use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    Expr, ExprKind, FormatAlignment, FormatArgPosition, FormatArgPositionKind, FormatArgs,
    FormatArgsPiece, FormatArgument, FormatArgumentKind, FormatArguments, FormatCount,
    FormatDebugHex, FormatOptions, FormatPlaceholder, FormatSign, FormatTrait,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, MultiSpan, PResult, SingleLabelManySpans};
use rustc_expand::base::{self, *};
use rustc_parse_format as parse;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{BytePos, InnerSpan, Span};

use rustc_lint_defs::builtin::NAMED_ARGUMENTS_USED_POSITIONALLY;
use rustc_lint_defs::{BufferedEarlyLint, BuiltinLintDiagnostics, LintId};

// The format_args!() macro is expanded in three steps:
//  1. First, `parse_args` will parse the `(literal, arg, arg, name=arg, name=arg)` syntax,
//     but doesn't parse the template (the literal) itself.
//  2. Second, `make_format_args` will parse the template, the format options, resolve argument references,
//     produce diagnostics, and turn the whole thing into a `FormatArgs` AST node.
//  3. Much later, in AST lowering (rustc_ast_lowering), that `FormatArgs` structure will be turned
//     into the expression of type `core::fmt::Arguments`.

// See rustc_ast/src/format.rs for the FormatArgs structure and glossary.

// Only used in parse_args and report_invalid_references,
// to indicate how a referred argument was used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositionUsedAs {
    Placeholder(Option<Span>),
    Precision,
    Width,
}
use PositionUsedAs::*;

use crate::errors;

struct MacroInput {
    fmtstr: P<Expr>,
    args: FormatArguments,
    /// Whether the first argument was a string literal or a result from eager macro expansion.
    /// If it's not a string literal, we disallow implicit argument capturing.
    ///
    /// This does not correspond to whether we can treat spans to the literal normally, as the whole
    /// invocation might be the result of another macro expansion, in which case this flag may still be true.
    ///
    /// See [RFC 2795] for more information.
    ///
    /// [RFC 2795]: https://rust-lang.github.io/rfcs/2795-format-args-implicit-identifiers.html#macro-hygiene
    is_direct_literal: bool,
}

/// Parses the arguments from the given list of tokens, returning the diagnostic
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the return value is:
///
/// ```text
/// Ok((fmtstr, parsed arguments))
/// ```
fn parse_args<'a>(ecx: &mut ExtCtxt<'a>, sp: Span, tts: TokenStream) -> PResult<'a, MacroInput> {
    let mut args = FormatArguments::new();

    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(ecx.create_err(errors::FormatRequiresString { span: sp }));
    }

    let first_token = &p.token;

    let fmtstr = if let token::Literal(lit) = first_token.kind && matches!(lit.kind, token::Str | token::StrRaw(_)) {
        // This allows us to properly handle cases when the first comma
        // after the format string is mistakenly replaced with any operator,
        // which cause the expression parser to eat too much tokens.
        p.parse_literal_maybe_minus()?
    } else {
        // Otherwise, we fall back to the expression parser.
        p.parse_expr()?
    };

    // Only allow implicit captures to be used when the argument is a direct literal
    // instead of a macro expanding to one.
    let is_direct_literal = matches!(fmtstr.kind, ExprKind::Lit(_));

    let mut first = true;

    while p.token != token::Eof {
        if !p.eat(&token::Comma) {
            if first {
                p.clear_expected_tokens();
            }

            match p.expect(&token::Comma) {
                Err(mut err) => {
                    match token::TokenKind::Comma.similar_tokens() {
                        Some(tks) if tks.contains(&p.token.kind) => {
                            // If a similar token is found, then it may be a typo. We
                            // consider it as a comma, and continue parsing.
                            err.emit();
                            p.bump();
                        }
                        // Otherwise stop the parsing and return the error.
                        _ => return Err(err),
                    }
                }
                Ok(recovered) => {
                    assert!(recovered);
                }
            }
        }
        first = false;
        if p.token == token::Eof {
            break;
        } // accept trailing commas
        match p.token.ident() {
            Some((ident, _)) if p.look_ahead(1, |t| *t == token::Eq) => {
                p.bump();
                p.expect(&token::Eq)?;
                let expr = p.parse_expr()?;
                if let Some((_, prev)) = args.by_name(ident.name) {
                    ecx.emit_err(errors::FormatDuplicateArg {
                        span: ident.span,
                        prev: prev.kind.ident().unwrap().span,
                        duplicate: ident.span,
                        ident,
                    });
                    continue;
                }
                args.add(FormatArgument { kind: FormatArgumentKind::Named(ident), expr });
            }
            _ => {
                let expr = p.parse_expr()?;
                if !args.named_args().is_empty() {
                    ecx.emit_err(errors::PositionalAfterNamed {
                        span: expr.span,
                        args: args
                            .named_args()
                            .iter()
                            .filter_map(|a| a.kind.ident().map(|ident| (a, ident)))
                            .map(|(arg, n)| n.span.to(arg.expr.span))
                            .collect(),
                    });
                }
                args.add(FormatArgument { kind: FormatArgumentKind::Normal, expr });
            }
        }
    }
    Ok(MacroInput { fmtstr, args, is_direct_literal })
}

fn make_format_args(
    ecx: &mut ExtCtxt<'_>,
    input: MacroInput,
    append_newline: bool,
) -> Result<FormatArgs, ()> {
    let msg = "format argument must be a string literal";
    let unexpanded_fmt_span = input.fmtstr.span;

    let MacroInput { fmtstr: efmt, mut args, is_direct_literal } = input;

    let (fmt_str, fmt_style, fmt_span) = match expr_to_spanned_string(ecx, efmt, msg) {
        Ok(mut fmt) if append_newline => {
            fmt.0 = Symbol::intern(&format!("{}\n", fmt.0));
            fmt
        }
        Ok(fmt) => fmt,
        Err(err) => {
            if let Some((mut err, suggested)) = err {
                let sugg_fmt = match args.explicit_args().len() {
                    0 => "{}".to_string(),
                    _ => format!("{}{{}}", "{} ".repeat(args.explicit_args().len())),
                };
                if !suggested {
                    err.span_suggestion(
                        unexpanded_fmt_span.shrink_to_lo(),
                        "you might be missing a string literal to format with",
                        format!("\"{}\", ", sugg_fmt),
                        Applicability::MaybeIncorrect,
                    );
                }
                err.emit();
            }
            return Err(());
        }
    };

    let str_style = match fmt_style {
        rustc_ast::StrStyle::Cooked => None,
        rustc_ast::StrStyle::Raw(raw) => Some(raw as usize),
    };

    let fmt_str = fmt_str.as_str(); // for the suggestions below
    let fmt_snippet = ecx.source_map().span_to_snippet(unexpanded_fmt_span).ok();
    let mut parser = parse::Parser::new(
        fmt_str,
        str_style,
        fmt_snippet,
        append_newline,
        parse::ParseMode::Format,
    );

    let mut pieces = Vec::new();
    while let Some(piece) = parser.next() {
        if !parser.errors.is_empty() {
            break;
        } else {
            pieces.push(piece);
        }
    }

    let is_source_literal = parser.is_source_literal;

    if !parser.errors.is_empty() {
        let err = parser.errors.remove(0);
        let sp = if is_source_literal {
            fmt_span.from_inner(InnerSpan::new(err.span.start, err.span.end))
        } else {
            // The format string could be another macro invocation, e.g.:
            //     format!(concat!("abc", "{}"), 4);
            // However, `err.span` is an inner span relative to the *result* of
            // the macro invocation, which is why we would get a nonsensical
            // result calling `fmt_span.from_inner(err.span)` as above, and
            // might even end up inside a multibyte character (issue #86085).
            // Therefore, we conservatively report the error for the entire
            // argument span here.
            fmt_span
        };
        let mut e = errors::InvalidFormatString {
            span: sp,
            note_: None,
            label_: None,
            sugg_: None,
            desc: err.description,
            label1: err.label,
        };
        if let Some(note) = err.note {
            e.note_ = Some(errors::InvalidFormatStringNote { note });
        }
        if let Some((label, span)) = err.secondary_label && is_source_literal {
            e.label_ = Some(errors::InvalidFormatStringLabel { span: fmt_span.from_inner(InnerSpan::new(span.start, span.end)), label } );
        }
        if err.should_be_replaced_with_positional_argument {
            let captured_arg_span =
                fmt_span.from_inner(InnerSpan::new(err.span.start, err.span.end));
            if let Ok(arg) = ecx.source_map().span_to_snippet(captured_arg_span) {
                let span = match args.unnamed_args().last() {
                    Some(arg) => arg.expr.span,
                    None => fmt_span,
                };
                e.sugg_ = Some(errors::InvalidFormatStringSuggestion {
                    captured: captured_arg_span,
                    len: args.unnamed_args().len().to_string(),
                    span: span.shrink_to_hi(),
                    arg,
                });
            }
        }
        ecx.emit_err(e);
        return Err(());
    }

    let to_span = |inner_span: rustc_parse_format::InnerSpan| {
        is_source_literal.then(|| {
            fmt_span.from_inner(InnerSpan { start: inner_span.start, end: inner_span.end })
        })
    };

    let mut used = vec![false; args.explicit_args().len()];
    let mut invalid_refs = Vec::new();
    let mut numeric_refences_to_named_arg = Vec::new();

    enum ArgRef<'a> {
        Index(usize),
        Name(&'a str, Option<Span>),
    }
    use ArgRef::*;

    let mut lookup_arg = |arg: ArgRef<'_>,
                          span: Option<Span>,
                          used_as: PositionUsedAs,
                          kind: FormatArgPositionKind|
     -> FormatArgPosition {
        let index = match arg {
            Index(index) => {
                if let Some(arg) = args.by_index(index) {
                    used[index] = true;
                    if arg.kind.ident().is_some() {
                        // This was a named argument, but it was used as a positional argument.
                        numeric_refences_to_named_arg.push((index, span, used_as));
                    }
                    Ok(index)
                } else {
                    // Doesn't exist as an explicit argument.
                    invalid_refs.push((index, span, used_as, kind));
                    Err(index)
                }
            }
            Name(name, span) => {
                let name = Symbol::intern(name);
                if let Some((index, _)) = args.by_name(name) {
                    // Name found in `args`, so we resolve it to its index.
                    if index < args.explicit_args().len() {
                        // Mark it as used, if it was an explicit argument.
                        used[index] = true;
                    }
                    Ok(index)
                } else {
                    // Name not found in `args`, so we add it as an implicitly captured argument.
                    let span = span.unwrap_or(fmt_span);
                    let ident = Ident::new(name, span);
                    let expr = if is_direct_literal {
                        ecx.expr_ident(span, ident)
                    } else {
                        // For the moment capturing variables from format strings expanded from macros is
                        // disabled (see RFC #2795)
                        ecx.emit_err(errors::FormatNoArgNamed { span, name });
                        DummyResult::raw_expr(span, true)
                    };
                    Ok(args.add(FormatArgument { kind: FormatArgumentKind::Captured(ident), expr }))
                }
            }
        };
        FormatArgPosition { index, kind, span }
    };

    let mut template = Vec::new();
    let mut unfinished_literal = String::new();
    let mut placeholder_index = 0;

    for piece in pieces {
        match piece {
            parse::Piece::String(s) => {
                unfinished_literal.push_str(s);
            }
            parse::Piece::NextArgument(box parse::Argument { position, position_span, format }) => {
                if !unfinished_literal.is_empty() {
                    template.push(FormatArgsPiece::Literal(Symbol::intern(&unfinished_literal)));
                    unfinished_literal.clear();
                }

                let span = parser.arg_places.get(placeholder_index).and_then(|&s| to_span(s));
                placeholder_index += 1;

                let position_span = to_span(position_span);
                let argument = match position {
                    parse::ArgumentImplicitlyIs(i) => lookup_arg(
                        Index(i),
                        position_span,
                        Placeholder(span),
                        FormatArgPositionKind::Implicit,
                    ),
                    parse::ArgumentIs(i) => lookup_arg(
                        Index(i),
                        position_span,
                        Placeholder(span),
                        FormatArgPositionKind::Number,
                    ),
                    parse::ArgumentNamed(name) => lookup_arg(
                        Name(name, position_span),
                        position_span,
                        Placeholder(span),
                        FormatArgPositionKind::Named,
                    ),
                };

                let alignment = match format.align {
                    parse::AlignUnknown => None,
                    parse::AlignLeft => Some(FormatAlignment::Left),
                    parse::AlignRight => Some(FormatAlignment::Right),
                    parse::AlignCenter => Some(FormatAlignment::Center),
                };

                let format_trait = match format.ty {
                    "" => FormatTrait::Display,
                    "?" => FormatTrait::Debug,
                    "e" => FormatTrait::LowerExp,
                    "E" => FormatTrait::UpperExp,
                    "o" => FormatTrait::Octal,
                    "p" => FormatTrait::Pointer,
                    "b" => FormatTrait::Binary,
                    "x" => FormatTrait::LowerHex,
                    "X" => FormatTrait::UpperHex,
                    _ => {
                        invalid_placeholder_type_error(ecx, format.ty, format.ty_span, fmt_span);
                        FormatTrait::Display
                    }
                };

                let precision_span = format.precision_span.and_then(to_span);
                let precision = match format.precision {
                    parse::CountIs(n) => Some(FormatCount::Literal(n)),
                    parse::CountIsName(name, name_span) => Some(FormatCount::Argument(lookup_arg(
                        Name(name, to_span(name_span)),
                        precision_span,
                        Precision,
                        FormatArgPositionKind::Named,
                    ))),
                    parse::CountIsParam(i) => Some(FormatCount::Argument(lookup_arg(
                        Index(i),
                        precision_span,
                        Precision,
                        FormatArgPositionKind::Number,
                    ))),
                    parse::CountIsStar(i) => Some(FormatCount::Argument(lookup_arg(
                        Index(i),
                        precision_span,
                        Precision,
                        FormatArgPositionKind::Implicit,
                    ))),
                    parse::CountImplied => None,
                };

                let width_span = format.width_span.and_then(to_span);
                let width = match format.width {
                    parse::CountIs(n) => Some(FormatCount::Literal(n)),
                    parse::CountIsName(name, name_span) => Some(FormatCount::Argument(lookup_arg(
                        Name(name, to_span(name_span)),
                        width_span,
                        Width,
                        FormatArgPositionKind::Named,
                    ))),
                    parse::CountIsParam(i) => Some(FormatCount::Argument(lookup_arg(
                        Index(i),
                        width_span,
                        Width,
                        FormatArgPositionKind::Number,
                    ))),
                    parse::CountIsStar(_) => unreachable!(),
                    parse::CountImplied => None,
                };

                template.push(FormatArgsPiece::Placeholder(FormatPlaceholder {
                    argument,
                    span,
                    format_trait,
                    format_options: FormatOptions {
                        fill: format.fill,
                        alignment,
                        sign: format.sign.map(|s| match s {
                            parse::Sign::Plus => FormatSign::Plus,
                            parse::Sign::Minus => FormatSign::Minus,
                        }),
                        alternate: format.alternate,
                        zero_pad: format.zero_pad,
                        debug_hex: format.debug_hex.map(|s| match s {
                            parse::DebugHex::Lower => FormatDebugHex::Lower,
                            parse::DebugHex::Upper => FormatDebugHex::Upper,
                        }),
                        precision,
                        width,
                    },
                }));
            }
        }
    }

    if !unfinished_literal.is_empty() {
        template.push(FormatArgsPiece::Literal(Symbol::intern(&unfinished_literal)));
    }

    if !invalid_refs.is_empty() {
        report_invalid_references(ecx, &invalid_refs, &template, fmt_span, &args, parser);
    }

    let unused = used
        .iter()
        .enumerate()
        .filter(|&(_, used)| !used)
        .map(|(i, _)| {
            let named = matches!(args.explicit_args()[i].kind, FormatArgumentKind::Named(_));
            (args.explicit_args()[i].expr.span, named)
        })
        .collect::<Vec<_>>();

    if !unused.is_empty() {
        // If there's a lot of unused arguments,
        // let's check if this format arguments looks like another syntax (printf / shell).
        let detect_foreign_fmt = unused.len() > args.explicit_args().len() / 2;
        report_missing_placeholders(ecx, unused, detect_foreign_fmt, str_style, fmt_str, fmt_span);
    }

    // Only check for unused named argument names if there are no other errors to avoid causing
    // too much noise in output errors, such as when a named argument is entirely unused.
    if invalid_refs.is_empty() && ecx.sess.err_count() == 0 {
        for &(index, span, used_as) in &numeric_refences_to_named_arg {
            let (position_sp_to_replace, position_sp_for_msg) = match used_as {
                Placeholder(pspan) => (span, pspan),
                Precision => {
                    // Strip the leading `.` for precision.
                    let span = span.map(|span| span.with_lo(span.lo() + BytePos(1)));
                    (span, span)
                }
                Width => (span, span),
            };
            let arg_name = args.explicit_args()[index].kind.ident().unwrap();
            ecx.buffered_early_lint.push(BufferedEarlyLint {
                span: arg_name.span.into(),
                msg: format!("named argument `{}` is not used by name", arg_name.name).into(),
                node_id: rustc_ast::CRATE_NODE_ID,
                lint_id: LintId::of(&NAMED_ARGUMENTS_USED_POSITIONALLY),
                diagnostic: BuiltinLintDiagnostics::NamedArgumentUsedPositionally {
                    position_sp_to_replace,
                    position_sp_for_msg,
                    named_arg_sp: arg_name.span,
                    named_arg_name: arg_name.name.to_string(),
                    is_formatting_arg: matches!(used_as, Width | Precision),
                },
            });
        }
    }

    Ok(FormatArgs { span: fmt_span, template, arguments: args })
}

fn invalid_placeholder_type_error(
    ecx: &ExtCtxt<'_>,
    ty: &str,
    ty_span: Option<rustc_parse_format::InnerSpan>,
    fmt_span: Span,
) {
    let sp = ty_span.map(|sp| fmt_span.from_inner(InnerSpan::new(sp.start, sp.end)));
    let suggs = if let Some(sp) = sp {
        [
            ("", "Display"),
            ("?", "Debug"),
            ("e", "LowerExp"),
            ("E", "UpperExp"),
            ("o", "Octal"),
            ("p", "Pointer"),
            ("b", "Binary"),
            ("x", "LowerHex"),
            ("X", "UpperHex"),
        ]
        .into_iter()
        .map(|(fmt, trait_name)| errors::FormatUnknownTraitSugg { span: sp, fmt, trait_name })
        .collect()
    } else {
        vec![]
    };
    ecx.emit_err(errors::FormatUnknownTrait { span: sp.unwrap_or(fmt_span), ty, suggs });
}

fn report_missing_placeholders(
    ecx: &mut ExtCtxt<'_>,
    unused: Vec<(Span, bool)>,
    detect_foreign_fmt: bool,
    str_style: Option<usize>,
    fmt_str: &str,
    fmt_span: Span,
) {
    let mut diag = if let &[(span, named)] = &unused[..] {
        ecx.create_err(errors::FormatUnusedArg { span, named })
    } else {
        let unused_labels =
            unused.iter().map(|&(span, named)| errors::FormatUnusedArg { span, named }).collect();
        let unused_spans = unused.iter().map(|&(span, _)| span).collect();
        ecx.create_err(errors::FormatUnusedArgs {
            fmt: fmt_span,
            unused: unused_spans,
            unused_labels,
        })
    };

    // Used to ensure we only report translations for *one* kind of foreign format.
    let mut found_foreign = false;

    // Decide if we want to look for foreign formatting directives.
    if detect_foreign_fmt {
        use super::format_foreign as foreign;

        // The set of foreign substitutions we've explained. This prevents spamming the user
        // with `%d should be written as {}` over and over again.
        let mut explained = FxHashSet::default();

        macro_rules! check_foreign {
            ($kind:ident) => {{
                let mut show_doc_note = false;

                let mut suggestions = vec![];
                // account for `"` and account for raw strings `r#`
                let padding = str_style.map(|i| i + 2).unwrap_or(1);
                for sub in foreign::$kind::iter_subs(fmt_str, padding) {
                    let (trn, success) = match sub.translate() {
                        Ok(trn) => (trn, true),
                        Err(Some(msg)) => (msg, false),

                        // If it has no translation, don't call it out specifically.
                        _ => continue,
                    };

                    let pos = sub.position();
                    let sub = String::from(sub.as_str());
                    if explained.contains(&sub) {
                        continue;
                    }
                    explained.insert(sub.clone());

                    if !found_foreign {
                        found_foreign = true;
                        show_doc_note = true;
                    }

                    if let Some(inner_sp) = pos {
                        let sp = fmt_span.from_inner(inner_sp);

                        if success {
                            suggestions.push((sp, trn));
                        } else {
                            diag.span_note(
                                sp,
                                format!("format specifiers use curly braces, and {}", trn),
                            );
                        }
                    } else {
                        if success {
                            diag.help(format!("`{}` should be written as `{}`", sub, trn));
                        } else {
                            diag.note(format!("`{}` should use curly braces, and {}", sub, trn));
                        }
                    }
                }

                if show_doc_note {
                    diag.note(concat!(
                        stringify!($kind),
                        " formatting is not supported; see the documentation for `std::fmt`",
                    ));
                }
                if suggestions.len() > 0 {
                    diag.multipart_suggestion(
                        "format specifiers use curly braces",
                        suggestions,
                        Applicability::MachineApplicable,
                    );
                }
            }};
        }

        check_foreign!(printf);
        if !found_foreign {
            check_foreign!(shell);
        }
    }
    if !found_foreign && unused.len() == 1 {
        diag.span_label(fmt_span, "formatting specifier missing");
    }

    diag.emit();
}

/// Handle invalid references to positional arguments. Output different
/// errors for the case where all arguments are positional and for when
/// there are named arguments or numbered positional arguments in the
/// format string.
fn report_invalid_references(
    ecx: &mut ExtCtxt<'_>,
    invalid_refs: &[(usize, Option<Span>, PositionUsedAs, FormatArgPositionKind)],
    template: &[FormatArgsPiece],
    fmt_span: Span,
    args: &FormatArguments,
    parser: parse::Parser<'_>,
) {
    let num_args_desc = match args.explicit_args().len() {
        0 => "no arguments were given".to_string(),
        1 => "there is 1 argument".to_string(),
        n => format!("there are {} arguments", n),
    };

    let mut e;

    if template.iter().all(|piece| match piece {
        FormatArgsPiece::Placeholder(FormatPlaceholder {
            argument: FormatArgPosition { kind: FormatArgPositionKind::Number, .. },
            ..
        }) => false,
        FormatArgsPiece::Placeholder(FormatPlaceholder {
            format_options:
                FormatOptions {
                    precision:
                        Some(FormatCount::Argument(FormatArgPosition {
                            kind: FormatArgPositionKind::Number,
                            ..
                        })),
                    ..
                }
                | FormatOptions {
                    width:
                        Some(FormatCount::Argument(FormatArgPosition {
                            kind: FormatArgPositionKind::Number,
                            ..
                        })),
                    ..
                },
            ..
        }) => false,
        _ => true,
    }) {
        // There are no numeric positions.
        // Collect all the implicit positions:
        let mut spans = Vec::new();
        let mut num_placeholders = 0;
        for piece in template {
            let mut placeholder = None;
            // `{arg:.*}`
            if let FormatArgsPiece::Placeholder(FormatPlaceholder {
                format_options:
                    FormatOptions {
                        precision:
                            Some(FormatCount::Argument(FormatArgPosition {
                                span,
                                kind: FormatArgPositionKind::Implicit,
                                ..
                            })),
                        ..
                    },
                ..
            }) = piece
            {
                placeholder = *span;
                num_placeholders += 1;
            }
            // `{}`
            if let FormatArgsPiece::Placeholder(FormatPlaceholder {
                argument: FormatArgPosition { kind: FormatArgPositionKind::Implicit, .. },
                span,
                ..
            }) = piece
            {
                placeholder = *span;
                num_placeholders += 1;
            }
            // For `{:.*}`, we only push one span.
            spans.extend(placeholder);
        }
        let span = if spans.is_empty() {
            MultiSpan::from_span(fmt_span)
        } else {
            MultiSpan::from_spans(spans)
        };
        e = ecx.create_err(errors::FormatPositionalMismatch {
            span,
            n: num_placeholders,
            desc: num_args_desc,
            highlight: SingleLabelManySpans {
                spans: args.explicit_args().iter().map(|arg| arg.expr.span).collect(),
                label: "",
                kind: rustc_errors::LabelKind::Label,
            },
        });
        // Point out `{:.*}` placeholders: those take an extra argument.
        let mut has_precision_star = false;
        for piece in template {
            if let FormatArgsPiece::Placeholder(FormatPlaceholder {
                format_options:
                    FormatOptions {
                        precision:
                            Some(FormatCount::Argument(FormatArgPosition {
                                index,
                                span: Some(span),
                                kind: FormatArgPositionKind::Implicit,
                                ..
                            })),
                        ..
                    },
                ..
            }) = piece
            {
                let (Ok(index) | Err(index)) = index;
                has_precision_star = true;
                e.span_label(
                    *span,
                    format!(
                        "this precision flag adds an extra required argument at position {}, which is why there {} expected",
                        index,
                        if num_placeholders == 1 {
                            "is 1 argument".to_string()
                        } else {
                            format!("are {} arguments", num_placeholders)
                        },
                    ),
                );
            }
        }
        if has_precision_star {
            e.note("positional arguments are zero-based");
        }
    } else {
        let mut indexes: Vec<_> = invalid_refs.iter().map(|&(index, _, _, _)| index).collect();
        // Avoid `invalid reference to positional arguments 7 and 7 (there is 1 argument)`
        // for `println!("{7:7$}", 1);`
        indexes.sort();
        indexes.dedup();
        let span: MultiSpan = if !parser.is_source_literal || parser.arg_places.is_empty() {
            MultiSpan::from_span(fmt_span)
        } else {
            MultiSpan::from_spans(invalid_refs.iter().filter_map(|&(_, span, _, _)| span).collect())
        };
        let arg_list = if let &[index] = &indexes[..] {
            format!("argument {index}")
        } else {
            let tail = indexes.pop().unwrap();
            format!(
                "arguments {head} and {tail}",
                head = indexes.into_iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
            )
        };
        e = ecx.struct_span_err(
            span,
            format!("invalid reference to positional {} ({})", arg_list, num_args_desc),
        );
        e.note("positional arguments are zero-based");
    }

    if template.iter().any(|piece| match piece {
        FormatArgsPiece::Placeholder(FormatPlaceholder { format_options: f, .. }) => {
            *f != FormatOptions::default()
        }
        _ => false,
    }) {
        e.note("for information about formatting flags, visit https://doc.rust-lang.org/std/fmt/index.html");
    }

    e.emit();
}

fn expand_format_args_impl<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    mut sp: Span,
    tts: TokenStream,
    nl: bool,
) -> Box<dyn base::MacResult + 'cx> {
    sp = ecx.with_def_site_ctxt(sp);
    match parse_args(ecx, sp, tts) {
        Ok(input) => {
            if let Ok(format_args) = make_format_args(ecx, input, nl) {
                MacEager::expr(ecx.expr(sp, ExprKind::FormatArgs(P(format_args))))
            } else {
                MacEager::expr(DummyResult::raw_expr(sp, true))
            }
        }
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}

pub fn expand_format_args<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    expand_format_args_impl(ecx, sp, tts, false)
}

pub fn expand_format_args_nl<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    expand_format_args_impl(ecx, sp, tts, true)
}
