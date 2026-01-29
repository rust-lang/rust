#![allow(warnings)]

use std::ops::Range;

use rustc_hir::attrs::diagnostic::{FormatArg, FormatString, Piece};
use rustc_hir::lints::FormatWarning;
use rustc_parse_format::{
    Argument, FormatSpec, ParseError, ParseMode, Parser, Piece as RpfPiece, Position,
};
use rustc_span::{InnerSpan, Span, Symbol, kw, sym};

pub mod on_unimplemented;

#[derive(Copy, Clone)]
pub(crate) enum Ctx {
    // `#[rustc_on_unimplemented]`
    RustcOnUnimplemented,
    // `#[diagnostic::...]`
    DiagnosticOnUnimplemented,
}

pub(crate) fn parse_format_string(
    input: Symbol,
    snippet: Option<String>,
    span: Span,
    ctx: Ctx,
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
                let arg = parse_arg(&arg, ctx, &mut warnings, span, parser.is_source_literal);
                Piece::Arg(arg)
            }
        })
        .collect();

    Ok((FormatString { input, pieces, span }, warnings))
}

fn parse_arg(
    arg: &Argument<'_>,
    ctx: Ctx,
    warnings: &mut Vec<FormatWarning>,
    input_span: Span,
    is_source_literal: bool,
) -> FormatArg {
    let span = slice_span(input_span, arg.position_span.clone(), is_source_literal);

    match arg.position {
        // Something like "hello {name}"
        Position::ArgumentNamed(name) => match (ctx, Symbol::intern(name)) {
            // Only `#[rustc_on_unimplemented]` can use these
            (Ctx::RustcOnUnimplemented { .. }, sym::ItemContext) => FormatArg::ItemContext,
            (Ctx::RustcOnUnimplemented { .. }, sym::This) => FormatArg::This,
            (Ctx::RustcOnUnimplemented { .. }, sym::Trait) => FormatArg::Trait,
            // Any attribute can use these
            (
                Ctx::RustcOnUnimplemented { .. } | Ctx::DiagnosticOnUnimplemented { .. },
                kw::SelfUpper,
            ) => FormatArg::SelfUpper,
            (
                Ctx::RustcOnUnimplemented { .. } | Ctx::DiagnosticOnUnimplemented { .. },
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
