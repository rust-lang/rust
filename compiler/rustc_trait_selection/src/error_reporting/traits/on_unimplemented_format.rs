use std::fmt;
use std::ops::Range;

use errors::*;
use rustc_middle::ty::print::TraitRefPrintSugared;
use rustc_middle::ty::{GenericParamDefKind, TyCtxt};
use rustc_parse_format::{
    Alignment, Argument, Count, FormatSpec, ParseError, ParseMode, Parser, Piece as RpfPiece,
    Position,
};
use rustc_session::lint::builtin::UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::def_id::DefId;
use rustc_span::{BytePos, Pos, Span, Symbol, kw, sym};

/// Like [std::fmt::Arguments] this is a string that has been parsed into "pieces",
/// either as string pieces or dynamic arguments.
#[derive(Debug)]
pub struct FormatString {
    #[allow(dead_code, reason = "Debug impl")]
    input: Symbol,
    span: Span,
    pieces: Vec<Piece>,
    /// The formatting string was parsed successfully but with warnings
    pub warnings: Vec<FormatWarning>,
}

#[derive(Debug)]
enum Piece {
    Lit(String),
    Arg(FormatArg),
}

#[derive(Debug)]
enum FormatArg {
    // A generic parameter, like `{T}` if we're on the `From<T>` trait.
    GenericParam {
        generic_param: Symbol,
    },
    // `{Self}`
    SelfUpper,
    /// `{This}` or `{TraitName}`
    This,
    /// The sugared form of the trait
    Trait,
    /// what we're in, like a function, method, closure etc.
    ItemContext,
    /// What the user typed, if it doesn't match anything we can use.
    AsIs(String),
}

pub enum Ctx<'tcx> {
    // `#[rustc_on_unimplemented]`
    RustcOnUnimplemented { tcx: TyCtxt<'tcx>, trait_def_id: DefId },
    // `#[diagnostic::...]`
    DiagnosticOnUnimplemented { tcx: TyCtxt<'tcx>, trait_def_id: DefId },
}

#[derive(Debug)]
pub enum FormatWarning {
    UnknownParam { argument_name: Symbol, span: Span },
    PositionalArgument { span: Span, help: String },
    InvalidSpecifier { name: String, span: Span },
    FutureIncompat { span: Span, help: String },
}

impl FormatWarning {
    pub fn emit_warning<'tcx>(&self, tcx: TyCtxt<'tcx>, item_def_id: DefId) {
        match *self {
            FormatWarning::UnknownParam { argument_name, span } => {
                let this = tcx.item_ident(item_def_id);
                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        span,
                        UnknownFormatParameterForOnUnimplementedAttr {
                            argument_name,
                            trait_name: this,
                        },
                    );
                }
            }
            FormatWarning::PositionalArgument { span, .. } => {
                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        span,
                        DisallowedPositionalArgument,
                    );
                }
            }
            FormatWarning::InvalidSpecifier { span, .. } => {
                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        span,
                        InvalidFormatSpecifier,
                    );
                }
            }
            FormatWarning::FutureIncompat { .. } => {
                // We've never deprecated anything in diagnostic namespace format strings
                // but if we do we will emit a warning here

                // FIXME(mejrs) in a couple releases, start emitting warnings for
                // #[rustc_on_unimplemented] deprecated args
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

impl FormatString {
    pub fn span(&self) -> Span {
        self.span
    }

    pub fn parse<'tcx>(
        input: Symbol,
        span: Span,
        ctx: &Ctx<'tcx>,
    ) -> Result<Self, Vec<ParseError>> {
        let s = input.as_str();
        let mut parser = Parser::new(s, None, None, false, ParseMode::Format);
        let mut pieces = Vec::new();
        let mut warnings = Vec::new();

        for piece in &mut parser {
            match piece {
                RpfPiece::Lit(lit) => {
                    pieces.push(Piece::Lit(lit.into()));
                }
                RpfPiece::NextArgument(arg) => {
                    warn_on_format_spec(arg.format.clone(), &mut warnings, span);
                    let arg = parse_arg(&arg, ctx, &mut warnings, span);
                    pieces.push(Piece::Arg(arg));
                }
            }
        }

        if parser.errors.is_empty() {
            Ok(FormatString { input, pieces, span, warnings })
        } else {
            Err(parser.errors)
        }
    }

    pub fn format(&self, args: &FormatArgs<'_>) -> String {
        let mut ret = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Lit(s) | Piece::Arg(FormatArg::AsIs(s)) => ret.push_str(&s),

                // `A` if we have `trait Trait<A> {}` and `note = "i'm the actual type of {A}"`
                Piece::Arg(FormatArg::GenericParam { generic_param }) => {
                    // Should always be some but we can't raise errors here
                    let value = match args.generic_args.iter().find(|(p, _)| p == generic_param) {
                        Some((_, val)) => val.to_string(),
                        None => generic_param.to_string(),
                    };
                    ret.push_str(&value);
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
}

fn parse_arg<'tcx>(
    arg: &Argument<'_>,
    ctx: &Ctx<'tcx>,
    warnings: &mut Vec<FormatWarning>,
    input_span: Span,
) -> FormatArg {
    let (Ctx::RustcOnUnimplemented { tcx, trait_def_id }
    | Ctx::DiagnosticOnUnimplemented { tcx, trait_def_id }) = ctx;

    let span = slice_span(input_span, arg.position_span.clone());

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
            ) if tcx.generics_of(trait_def_id).own_params.iter().any(|param| {
                !matches!(param.kind, GenericParamDefKind::Lifetime) && param.name == generic_param
            }) =>
            {
                FormatArg::GenericParam { generic_param }
            }

            (_, argument_name) => {
                warnings.push(FormatWarning::UnknownParam { argument_name, span });
                FormatArg::AsIs(format!("{{{}}}", argument_name.as_str()))
            }
        },

        // `{:1}` and `{}` are ignored
        Position::ArgumentIs(idx) => {
            warnings.push(FormatWarning::PositionalArgument {
                span,
                help: format!("use `{{{idx}}}` to print a number in braces"),
            });
            FormatArg::AsIs(format!("{{{idx}}}"))
        }
        Position::ArgumentImplicitlyIs(_) => {
            warnings.push(FormatWarning::PositionalArgument {
                span,
                help: String::from("use `{{}}` to print empty braces"),
            });
            FormatArg::AsIs(String::from("{}"))
        }
    }
}

/// `#[rustc_on_unimplemented]` and `#[diagnostic::...]` don't actually do anything
/// with specifiers, so emit a warning if they are used.
fn warn_on_format_spec(spec: FormatSpec<'_>, warnings: &mut Vec<FormatWarning>, input_span: Span) {
    if !matches!(
        spec,
        FormatSpec {
            fill: None,
            fill_span: None,
            align: Alignment::AlignUnknown,
            sign: None,
            alternate: false,
            zero_pad: false,
            debug_hex: None,
            precision: Count::CountImplied,
            precision_span: None,
            width: Count::CountImplied,
            width_span: None,
            ty: _,
            ty_span: _,
        },
    ) {
        let span = spec.ty_span.map(|inner| slice_span(input_span, inner)).unwrap_or(input_span);
        warnings.push(FormatWarning::InvalidSpecifier { span, name: spec.ty.into() })
    }
}

fn slice_span(input: Span, range: Range<usize>) -> Span {
    let span = input.data();

    Span::new(
        span.lo + BytePos::from_usize(range.start),
        span.lo + BytePos::from_usize(range.end),
        span.ctxt,
        span.parent,
    )
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
