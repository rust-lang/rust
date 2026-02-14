#![allow(warnings)]
use std::ops::Range;

use rustc_errors::E0232;
use rustc_hir::AttrPath;
use rustc_hir::attrs::diagnostic::{
    FilterFormatString, Flag, FormatArg, FormatString, LitOrArg, Name, NameValue,
    OnUnimplementedCondition, Piece, Predicate,
};
use rustc_hir::lints::FormatWarning;
use rustc_macros::Diagnostic;
use rustc_parse_format::{
    Argument, FormatSpec, ParseError, ParseMode, Parser, Piece as RpfPiece, Position,
};
use rustc_span::{Ident, InnerSpan, Span, Symbol, kw, sym};
use thin_vec::ThinVec;

use crate::parser::{ArgParser, MetaItemListParser, MetaItemOrLitParser, MetaItemParser};

pub mod on_unimplemented;

#[derive(Copy, Clone)]
pub(crate) enum Mode {
    // `#[rustc_on_unimplemented]`
    RustcOnUnimplemented,
    // `#[diagnostic::...]`
    DiagnosticOnUnimplemented,
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
            (
                Mode::RustcOnUnimplemented { .. } | Mode::DiagnosticOnUnimplemented { .. },
                kw::SelfUpper,
            ) => FormatArg::SelfUpper,
            (
                Mode::RustcOnUnimplemented { .. } | Mode::DiagnosticOnUnimplemented { .. },
                generic_param,
            ) => FormatArg::GenericParam { generic_param, span },
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
            let name = parse_name(predicate);
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

fn parse_name(Ident { name, span }: Ident) -> Name {
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
                Position::ArgumentNamed(arg @ "integer" | arg @ "integral" | arg @ "float") => {
                    LitOrArg::Lit(Symbol::intern(&format!("{{{arg}}}")))
                }

                // FIXME(mejrs) We should check if these correspond to a generic of the trait.
                Position::ArgumentNamed(arg) => LitOrArg::Arg(Symbol::intern(arg)),

                // FIXME(mejrs) These should really be warnings/errors
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
    #[diag("invalid name in `on`-clause", code = E0232)]
    InvalidName {
        #[primary_span]
        #[label(
            "expected one of `cause`, `from_desugaring`, `Self` or any generic parameter of the trait, not `{$invalid_name}`"
        )]
        span: Span,
        invalid_name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("this attribute must have a value", code = E0232)]
#[note("e.g. `#[rustc_on_unimplemented(message=\"foo\")]`")]
pub struct NoValueInOnUnimplemented {
    #[primary_span]
    #[label("expected value here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "using multiple `rustc_on_unimplemented` (or mixing it with `diagnostic::on_unimplemented`) is not supported"
)]
pub struct DupesNotAllowed;
