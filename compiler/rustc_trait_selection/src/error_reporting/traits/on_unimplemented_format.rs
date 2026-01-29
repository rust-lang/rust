use std::fmt;
use std::ops::Range;

use rustc_hir::attrs::diagnostic::*;
use rustc_hir::lints::FormatWarning;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::TraitRefPrintSugared;
use rustc_parse_format::{
    Argument, FormatSpec, ParseError, ParseMode, Parser, Piece as RpfPiece, Position,
};
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_FORMAT_LITERALS;
use rustc_span::def_id::DefId;
use rustc_span::{InnerSpan, Span, Symbol, kw, sym};

use crate::error_reporting::traits::on_unimplemented_format::errors::*;
pub enum Ctx<'tcx> {
    // `#[rustc_on_unimplemented]`
    RustcOnUnimplemented { tcx: TyCtxt<'tcx>, trait_def_id: DefId },
    // `#[diagnostic::...]`
    DiagnosticOnUnimplemented { tcx: TyCtxt<'tcx>, trait_def_id: DefId },
}

pub fn emit_warning<'tcx>(slf: &FormatWarning, tcx: TyCtxt<'tcx>, item_def_id: DefId) {
    match *slf {
        FormatWarning::PositionalArgument { span, .. } => {
            if let Some(item_def_id) = item_def_id.as_local() {
                tcx.emit_node_span_lint(
                    MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                    tcx.local_def_id_to_hir_id(item_def_id),
                    span,
                    DisallowedPositionalArgument,
                );
            }
        }
        FormatWarning::InvalidSpecifier { span, .. } => {
            if let Some(item_def_id) = item_def_id.as_local() {
                tcx.emit_node_span_lint(
                    MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                    tcx.local_def_id_to_hir_id(item_def_id),
                    span,
                    InvalidFormatSpecifier,
                );
            }
        }
    }
}

/// Arguments to fill a [FormatString] with.
///
/// For example, given a
/// ```rust,ignore (just an example)
///
/// #[rustc_on_unimplemented(
///     on(all(from_desugaring = "QuestionMark"),
///         message = "the `?` operator can only be used in {ItemContext} \
///                     that returns `Result` or `Option` \
///                     (or another type that implements `{FromResidual}`)",
///         label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
///         parent_label = "this function should return `Result` or `Option` to accept `?`"
///     ),
/// )]
/// pub trait FromResidual<R = <Self as Try>::Residual> {
///    ...
/// }
///
/// async fn an_async_function() -> u32 {
///     let x: Option<u32> = None;
///     x?; //~ ERROR the `?` operator
///     22
/// }
///  ```
/// it will look like this:
///
/// ```rust,ignore (just an example)
/// FormatArgs {
///     this: "FromResidual",
///     trait_sugared: "FromResidual<Option<Infallible>>",
///     item_context: "an async function",
///     generic_args: [("Self", "u32"), ("R", "Option<Infallible>")],
/// }
/// ```
#[derive(Debug)]
pub struct FormatArgs<'tcx> {
    pub this: String,
    pub trait_sugared: TraitRefPrintSugared<'tcx>,
    pub item_context: &'static str,
    pub generic_args: Vec<(Symbol, String)>,
}

pub fn parse_format_string<'tcx>(
    input: Symbol,
    snippet: Option<String>,
    span: Span,
    ctx: &Ctx<'tcx>,
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

pub fn format(slf: &FormatString, args: &FormatArgs<'_>) -> String {
    let mut ret = String::new();
    for piece in &slf.pieces {
        match piece {
            Piece::Lit(s) | Piece::Arg(FormatArg::AsIs(s)) => ret.push_str(s.as_str()),

            // `A` if we have `trait Trait<A> {}` and `note = "i'm the actual type of {A}"`
            Piece::Arg(FormatArg::GenericParam { generic_param, .. }) => {
                match args.generic_args.iter().find(|(p, _)| p == generic_param) {
                    Some((_, val)) => ret.push_str(val.as_str()),

                    None => {
                        // Apparently this was not actually a generic parameter, so lets write
                        // what the user wrote.
                        let _ = fmt::write(&mut ret, format_args!("{{{generic_param}}}"));
                    }
                }
            }
            // `{Self}`
            Piece::Arg(FormatArg::SelfUpper) => {
                let slf = match args.generic_args.iter().find(|(p, _)| *p == kw::SelfUpper) {
                    Some((_, val)) => val.to_string(),
                    None => "Self".to_string(),
                };
                ret.push_str(&slf);
            }

            // It's only `rustc_onunimplemented` from here
            Piece::Arg(FormatArg::This) => ret.push_str(&args.this),
            Piece::Arg(FormatArg::Trait) => {
                let _ = fmt::write(&mut ret, format_args!("{}", &args.trait_sugared));
            }
            Piece::Arg(FormatArg::ItemContext) => ret.push_str(args.item_context),
        }
    }
    ret
}

fn parse_arg<'tcx>(
    arg: &Argument<'_>,
    ctx: &Ctx<'tcx>,
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

pub mod errors {
    use rustc_macros::LintDiagnostic;
    use rustc_span::Ident;

    use super::*;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_unknown_format_parameter_for_on_unimplemented_attr)]
    #[help]
    pub struct UnknownFormatParameterForOnUnimplementedAttr {
        pub argument_name: Symbol,
        pub trait_name: Ident,
    }

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_disallowed_positional_argument)]
    #[help]
    pub struct DisallowedPositionalArgument;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_invalid_format_specifier)]
    #[help]
    pub struct InvalidFormatSpecifier;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_missing_options_for_on_unimplemented_attr)]
    #[help]
    pub struct MissingOptionsForOnUnimplementedAttr;
}
