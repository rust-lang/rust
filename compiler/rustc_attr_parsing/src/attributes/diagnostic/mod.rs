use std::ops::Range;

use rustc_errors::E0232;
use rustc_hir::AttrPath;
use rustc_hir::attrs::diagnostic::{
    AppendConstMessage, Directive, FilterFormatString, Flag, FormatArg, FormatString, LitOrArg,
    Name, NameValue, OnUnimplementedCondition, Piece, Predicate,
};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
use rustc_macros::Diagnostic;
use rustc_parse_format::{
    Argument, FormatSpec, ParseError, ParseMode, Parser, Piece as RpfPiece, Position,
};
use rustc_session::lint::builtin::{
    MALFORMED_DIAGNOSTIC_ATTRIBUTES, MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
};
use rustc_span::{Ident, InnerSpan, Span, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::context::{AcceptContext, Stage};
use crate::parser::{ArgParser, MetaItemListParser, MetaItemOrLitParser, MetaItemParser};

pub(crate) mod do_not_recommend;
pub(crate) mod on_const;
pub(crate) mod on_unimplemented;

#[derive(Copy, Clone)]
pub(crate) enum Mode {
    /// `#[rustc_on_unimplemented]`
    RustcOnUnimplemented,
    /// `#[diagnostic::on_unimplemented]`
    DiagnosticOnUnimplemented,
    /// `#[diagnostic::on_const]`
    DiagnosticOnConst,
}

fn merge_directives<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    first: &mut Option<(Span, Directive)>,
    later: (Span, Directive),
) {
    if let Some((_, first)) = first {
        if first.is_rustc_attr || later.1.is_rustc_attr {
            cx.emit_err(DupesNotAllowed);
        }

        merge(cx, &mut first.message, later.1.message, sym::message);
        merge(cx, &mut first.label, later.1.label, sym::label);
        first.notes.extend(later.1.notes);
    } else {
        *first = Some(later);
    }
}

fn merge<T, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    first: &mut Option<(Span, T)>,
    later: Option<(Span, T)>,
    option_name: Symbol,
) {
    match (first, later) {
        (Some(_) | None, None) => {}
        (Some((first_span, _)), Some((later_span, _))) => {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::IgnoredDiagnosticOption {
                    first_span: *first_span,
                    later_span,
                    option_name,
                },
                later_span,
            );
        }
        (first @ None, Some(later)) => {
            first.get_or_insert(later);
        }
    }
}

fn parse_directive_items<'p, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    mode: Mode,
    items: impl Iterator<Item = &'p MetaItemOrLitParser>,
    is_root: bool,
) -> Option<Directive> {
    let condition = None;
    let mut message: Option<(Span, _)> = None;
    let mut label: Option<(Span, _)> = None;
    let mut notes = ThinVec::new();
    let mut parent_label = None;
    let mut subcommands = ThinVec::new();
    let mut append_const_msg = None;

    for item in items {
        let span = item.span();

        macro malformed() {{
            match mode {
                Mode::RustcOnUnimplemented => {
                    cx.emit_err(NoValueInOnUnimplemented { span: item.span() });
                }
                Mode::DiagnosticOnUnimplemented => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnUnimplementedAttr { span },
                        span,
                    );
                }
                Mode::DiagnosticOnConst => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnConstAttr { span },
                        span,
                    );
                }
            }
            continue;
        }}

        macro or_malformed($($code:tt)*) {{
            let Some(ret) = (||{
                Some($($code)*)
            })() else {

                malformed!()
            };
            ret
        }}

        macro duplicate($name: ident, $($first_span:tt)*) {{
            match mode {
                Mode::RustcOnUnimplemented => {
                    cx.emit_err(NoValueInOnUnimplemented { span: item.span() });
                }
                Mode::DiagnosticOnUnimplemented |Mode::DiagnosticOnConst => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::IgnoredDiagnosticOption {
                            first_span: $($first_span)*,
                            later_span: span,
                            option_name: $name,
                        },
                        span,
                    );
                }
            }
        }}

        let item: &MetaItemParser = or_malformed!(item.meta_item()?);
        let name = or_malformed!(item.ident()?).name;

        // Some things like `message = "message"` must have a value.
        // But with things like `append_const_msg` that is optional.
        let value: Option<Ident> = match item.args().name_value() {
            Some(nv) => Some(or_malformed!(nv.value_as_ident()?)),
            None => None,
        };

        let mut parse_format = |input: Ident| {
            let snippet = cx.sess.source_map().span_to_snippet(input.span).ok();
            let is_snippet = snippet.is_some();
            match parse_format_string(input.name, snippet, input.span, mode) {
                Ok((f, warnings)) => {
                    for warning in warnings {
                        let (FormatWarning::InvalidSpecifier { span, .. }
                        | FormatWarning::PositionalArgument { span, .. }) = warning;
                        cx.emit_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            AttributeLintKind::MalformedDiagnosticFormat { warning },
                            span,
                        );
                    }

                    f
                }
                Err(e) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                        AttributeLintKind::DiagnosticWrappedParserError {
                            description: e.description,
                            label: e.label,
                            span: slice_span(input.span, e.span, is_snippet),
                        },
                        input.span,
                    );
                    // We could not parse the input, just use it as-is.
                    FormatString {
                        input: input.name,
                        span: input.span,
                        pieces: thin_vec![Piece::Lit(input.name)],
                    }
                }
            }
        };
        match (mode, name) {
            (_, sym::message) => {
                let value = or_malformed!(value?);
                if let Some(message) = &message {
                    duplicate!(name, message.0)
                } else {
                    message = Some((item.span(), parse_format(value)));
                }
            }
            (_, sym::label) => {
                let value = or_malformed!(value?);
                if let Some(label) = &label {
                    duplicate!(name, label.0)
                } else {
                    label = Some((item.span(), parse_format(value)));
                }
            }
            (_, sym::note) => {
                let value = or_malformed!(value?);
                notes.push(parse_format(value))
            }

            (Mode::RustcOnUnimplemented, sym::append_const_msg) => {
                append_const_msg = if let Some(msg) = value {
                    Some(AppendConstMessage::Custom(msg.name, item.span()))
                } else {
                    Some(AppendConstMessage::Default)
                }
            }
            (Mode::RustcOnUnimplemented, sym::parent_label) => {
                let value = or_malformed!(value?);
                if parent_label.is_none() {
                    parent_label = Some(parse_format(value));
                } else {
                    duplicate!(name, span)
                }
            }
            (Mode::RustcOnUnimplemented, sym::on) => {
                if is_root {
                    let items = or_malformed!(item.args().list()?);
                    let mut iter = items.mixed();
                    let condition: &MetaItemOrLitParser = match iter.next() {
                        Some(c) => c,
                        None => {
                            cx.emit_err(InvalidOnClause::Empty { span });
                            continue;
                        }
                    };

                    let condition = parse_condition(condition);

                    if items.len() < 2 {
                        // Something like `#[rustc_on_unimplemented(on(.., /* nothing */))]`
                        // There's a condition but no directive behind it, this is a mistake.
                        malformed!();
                    }

                    let mut directive =
                        or_malformed!(parse_directive_items(cx, mode, iter, false)?);

                    match condition {
                        Ok(c) => {
                            directive.condition = Some(c);
                            subcommands.push(directive);
                        }
                        Err(e) => {
                            cx.emit_err(e);
                        }
                    }
                } else {
                    malformed!();
                }
            }

            _other => {
                malformed!();
            }
        }
    }

    Some(Directive {
        is_rustc_attr: matches!(mode, Mode::RustcOnUnimplemented),
        condition,
        subcommands,
        message,
        label,
        notes,
        parent_label,
        append_const_msg,
    })
}

pub(crate) fn parse_format_string(
    input: Symbol,
    snippet: Option<String>,
    span: Span,
    mode: Mode,
) -> Result<(FormatString, Vec<FormatWarning>), ParseError> {
    let s = input.as_str();
    let mut parser = Parser::new(s, None, snippet, false, ParseMode::Diagnostic);
    let pieces: Vec<_> = parser.by_ref().collect();

    if let Some(err) = parser.errors.into_iter().next() {
        return Err(err);
    }
    let mut warnings = Vec::new();

    let pieces = pieces
        .into_iter()
        .map(|piece| match piece {
            RpfPiece::Lit(lit) => Piece::Lit(Symbol::intern(lit)),
            RpfPiece::NextArgument(arg) => {
                warn_on_format_spec(&arg.format, &mut warnings, span, parser.is_source_literal);
                let arg = parse_arg(&arg, mode, &mut warnings, span, parser.is_source_literal);
                Piece::Arg(arg)
            }
        })
        .collect();

    Ok((FormatString { input, pieces, span }, warnings))
}

fn parse_arg(
    arg: &Argument<'_>,
    mode: Mode,
    warnings: &mut Vec<FormatWarning>,
    input_span: Span,
    is_source_literal: bool,
) -> FormatArg {
    let span = slice_span(input_span, arg.position_span.clone(), is_source_literal);

    match arg.position {
        // Something like "hello {name}"
        Position::ArgumentNamed(name) => match (mode, Symbol::intern(name)) {
            // Only `#[rustc_on_unimplemented]` can use these
            (Mode::RustcOnUnimplemented { .. }, sym::ItemContext) => FormatArg::ItemContext,
            (Mode::RustcOnUnimplemented { .. }, sym::This) => FormatArg::This,
            (Mode::RustcOnUnimplemented { .. }, sym::Trait) => FormatArg::Trait,
            // Any attribute can use these
            (_, kw::SelfUpper) => FormatArg::SelfUpper,
            (_, generic_param) => FormatArg::GenericParam { generic_param, span },
        },

        // `{:1}` and `{}` are ignored
        Position::ArgumentIs(idx) => {
            warnings.push(FormatWarning::PositionalArgument {
                span,
                help: format!("use `{{{idx}}}` to print a number in braces"),
            });
            FormatArg::AsIs(Symbol::intern(&format!("{{{idx}}}")))
        }
        Position::ArgumentImplicitlyIs(_) => {
            warnings.push(FormatWarning::PositionalArgument {
                span,
                help: String::from("use `{{}}` to print empty braces"),
            });
            FormatArg::AsIs(sym::empty_braces)
        }
    }
}

/// `#[rustc_on_unimplemented]` and `#[diagnostic::...]` don't actually do anything
/// with specifiers, so emit a warning if they are used.
fn warn_on_format_spec(
    spec: &FormatSpec<'_>,
    warnings: &mut Vec<FormatWarning>,
    input_span: Span,
    is_source_literal: bool,
) {
    if spec.ty != "" {
        let span = spec
            .ty_span
            .as_ref()
            .map(|inner| slice_span(input_span, inner.clone(), is_source_literal))
            .unwrap_or(input_span);
        warnings.push(FormatWarning::InvalidSpecifier { span, name: spec.ty.into() })
    }
}

fn slice_span(input: Span, Range { start, end }: Range<usize>, is_source_literal: bool) -> Span {
    if is_source_literal { input.from_inner(InnerSpan { start, end }) } else { input }
}

pub(crate) fn parse_condition(
    input: &MetaItemOrLitParser,
) -> Result<OnUnimplementedCondition, InvalidOnClause> {
    let span = input.span();
    let pred = parse_predicate(input)?;
    Ok(OnUnimplementedCondition { span, pred })
}

fn parse_predicate(input: &MetaItemOrLitParser) -> Result<Predicate, InvalidOnClause> {
    let Some(meta_item) = input.meta_item() else {
        return Err(InvalidOnClause::UnsupportedLiteral { span: input.span() });
    };

    let Some(predicate) = meta_item.ident() else {
        return Err(InvalidOnClause::ExpectedIdentifier {
            span: meta_item.path().span(),
            path: meta_item.path().get_attribute_path(),
        });
    };

    match meta_item.args() {
        ArgParser::List(mis) => match predicate.name {
            sym::any => Ok(Predicate::Any(parse_predicate_sequence(mis)?)),
            sym::all => Ok(Predicate::All(parse_predicate_sequence(mis)?)),
            sym::not => {
                if let Some(single) = mis.single() {
                    Ok(Predicate::Not(Box::new(parse_predicate(single)?)))
                } else {
                    Err(InvalidOnClause::ExpectedOnePredInNot { span: mis.span })
                }
            }
            invalid_pred => {
                Err(InvalidOnClause::InvalidPredicate { span: predicate.span, invalid_pred })
            }
        },
        ArgParser::NameValue(p) => {
            let Some(value) = p.value_as_ident() else {
                return Err(InvalidOnClause::UnsupportedLiteral { span: p.args_span() });
            };
            let name = parse_name(predicate.name);
            let value = parse_filter(value.name);
            let kv = NameValue { name, value };
            Ok(Predicate::Match(kv))
        }
        ArgParser::NoArgs => {
            let flag = parse_flag(predicate)?;
            Ok(Predicate::Flag(flag))
        }
    }
}

fn parse_predicate_sequence(
    sequence: &MetaItemListParser,
) -> Result<ThinVec<Predicate>, InvalidOnClause> {
    sequence.mixed().map(parse_predicate).collect()
}

fn parse_flag(Ident { name, span }: Ident) -> Result<Flag, InvalidOnClause> {
    match name {
        sym::crate_local => Ok(Flag::CrateLocal),
        sym::direct => Ok(Flag::Direct),
        sym::from_desugaring => Ok(Flag::FromDesugaring),
        invalid_flag => Err(InvalidOnClause::InvalidFlag { invalid_flag, span }),
    }
}

fn parse_name(name: Symbol) -> Name {
    match name {
        kw::SelfUpper => Name::SelfUpper,
        sym::from_desugaring => Name::FromDesugaring,
        sym::cause => Name::Cause,
        generic => Name::GenericArg(generic),
    }
}

fn parse_filter(input: Symbol) -> FilterFormatString {
    let pieces = Parser::new(input.as_str(), None, None, false, ParseMode::Diagnostic)
        .map(|p| match p {
            RpfPiece::Lit(s) => LitOrArg::Lit(Symbol::intern(s)),
            // We just ignore formatspecs here
            RpfPiece::NextArgument(a) => match a.position {
                // In `TypeErrCtxt::on_unimplemented_note` we substitute `"{integral}"` even
                // if the integer type has been resolved, to allow targeting all integers.
                // `"{integer}"` and `"{float}"` come from numerics that haven't been inferred yet,
                // from the `Display` impl of `InferTy` to be precise.
                //
                // Don't try to format these later!
                Position::ArgumentNamed(arg @ ("integer" | "integral" | "float")) => {
                    LitOrArg::Lit(Symbol::intern(&format!("{{{arg}}}")))
                }

                Position::ArgumentNamed(arg) => LitOrArg::Arg(Symbol::intern(arg)),
                Position::ArgumentImplicitlyIs(_) => LitOrArg::Lit(sym::empty_braces),
                Position::ArgumentIs(idx) => LitOrArg::Lit(Symbol::intern(&format!("{{{idx}}}"))),
            },
        })
        .collect();
    FilterFormatString { pieces }
}

#[derive(Diagnostic)]
pub(crate) enum InvalidOnClause {
    #[diag("empty `on`-clause in `#[rustc_on_unimplemented]`", code = E0232)]
    Empty {
        #[primary_span]
        #[label("empty `on`-clause here")]
        span: Span,
    },
    #[diag("expected a single predicate in `not(..)`", code = E0232)]
    ExpectedOnePredInNot {
        #[primary_span]
        #[label("unexpected quantity of predicates here")]
        span: Span,
    },
    #[diag("literals inside `on`-clauses are not supported", code = E0232)]
    UnsupportedLiteral {
        #[primary_span]
        #[label("unexpected literal here")]
        span: Span,
    },
    #[diag("expected an identifier inside this `on`-clause", code = E0232)]
    ExpectedIdentifier {
        #[primary_span]
        #[label("expected an identifier here, not `{$path}`")]
        span: Span,
        path: AttrPath,
    },
    #[diag("this predicate is invalid", code = E0232)]
    InvalidPredicate {
        #[primary_span]
        #[label("expected one of `any`, `all` or `not` here, not `{$invalid_pred}`")]
        span: Span,
        invalid_pred: Symbol,
    },
    #[diag("invalid flag in `on`-clause", code = E0232)]
    InvalidFlag {
        #[primary_span]
        #[label(
            "expected one of the `crate_local`, `direct` or `from_desugaring` flags, not `{$invalid_flag}`"
        )]
        span: Span,
        invalid_flag: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("this attribute must have a value", code = E0232)]
#[note("e.g. `#[rustc_on_unimplemented(message=\"foo\")]`")]
pub(crate) struct NoValueInOnUnimplemented {
    #[primary_span]
    #[label("expected value here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "using multiple `rustc_on_unimplemented` (or mixing it with `diagnostic::on_unimplemented`) is not supported"
)]
pub(crate) struct DupesNotAllowed;
