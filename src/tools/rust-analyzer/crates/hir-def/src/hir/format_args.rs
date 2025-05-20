//! Parses `format_args` input.

use either::Either;
use hir_expand::name::Name;
use intern::Symbol;
use rustc_parse_format as parse;
use span::SyntaxContext;
use stdx::TupleExt;
use syntax::{
    TextRange,
    ast::{self, IsString},
};

use crate::hir::ExprId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatArgs {
    pub template: Box<[FormatArgsPiece]>,
    pub arguments: FormatArguments,
    pub orphans: Vec<ExprId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatArguments {
    pub arguments: Box<[FormatArgument]>,
    pub num_unnamed_args: usize,
    pub num_explicit_args: usize,
    pub names: Box<[(Name, usize)]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatArgsPiece {
    Literal(Symbol),
    Placeholder(FormatPlaceholder),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatPlaceholder {
    /// Index into [`FormatArgs::arguments`].
    pub argument: FormatArgPosition,
    /// The span inside the format string for the full `{…}` placeholder.
    pub span: Option<TextRange>,
    /// `{}`, `{:?}`, or `{:x}`, etc.
    pub format_trait: FormatTrait,
    /// `{}` or `{:.5}` or `{:-^20}`, etc.
    pub format_options: FormatOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatArgPosition {
    /// Which argument this position refers to (Ok),
    /// or would've referred to if it existed (Err).
    pub index: Result<usize, Either<usize, Name>>,
    /// What kind of position this is. See [`FormatArgPositionKind`].
    pub kind: FormatArgPositionKind,
    /// The span of the name or number.
    pub span: Option<TextRange>,
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum FormatArgPositionKind {
    /// `{}` or `{:.*}`
    Implicit,
    /// `{1}` or `{:1$}` or `{:.1$}`
    Number,
    /// `{a}` or `{:a$}` or `{:.a$}`
    Named,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum FormatTrait {
    /// `{}`
    Display,
    /// `{:?}`
    Debug,
    /// `{:e}`
    LowerExp,
    /// `{:E}`
    UpperExp,
    /// `{:o}`
    Octal,
    /// `{:p}`
    Pointer,
    /// `{:b}`
    Binary,
    /// `{:x}`
    LowerHex,
    /// `{:X}`
    UpperHex,
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct FormatOptions {
    /// The width. E.g. `{:5}` or `{:width$}`.
    pub width: Option<FormatCount>,
    /// The precision. E.g. `{:.5}` or `{:.precision$}`.
    pub precision: Option<FormatCount>,
    /// The alignment. E.g. `{:>}` or `{:<}` or `{:^}`.
    pub alignment: Option<FormatAlignment>,
    /// The fill character. E.g. the `.` in `{:.>10}`.
    pub fill: Option<char>,
    /// The `+` or `-` flag.
    pub sign: Option<FormatSign>,
    /// The `#` flag.
    pub alternate: bool,
    /// The `0` flag. E.g. the `0` in `{:02x}`.
    pub zero_pad: bool,
    /// The `x` or `X` flag (for `Debug` only). E.g. the `x` in `{:x?}`.
    pub debug_hex: Option<FormatDebugHex>,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FormatSign {
    /// The `+` flag.
    Plus,
    /// The `-` flag.
    Minus,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FormatDebugHex {
    /// The `x` flag in `{:x?}`.
    Lower,
    /// The `X` flag in `{:X?}`.
    Upper,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FormatAlignment {
    /// `{:<}`
    Left,
    /// `{:>}`
    Right,
    /// `{:^}`
    Center,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatCount {
    /// `{:5}` or `{:.5}`
    Literal(u16),
    /// `{:.*}`, `{:.5$}`, or `{:a$}`, etc.
    Argument(FormatArgPosition),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatArgument {
    pub kind: FormatArgumentKind,
    pub expr: ExprId,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FormatArgumentKind {
    /// `format_args(…, arg)`
    Normal,
    /// `format_args(…, arg = 1)`
    Named(Name),
    /// `format_args("… {arg} …")`
    Captured(Name),
}

// Only used in parse_args and report_invalid_references,
// to indicate how a referred argument was used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositionUsedAs {
    Placeholder(Option<TextRange>),
    Precision,
    Width,
}
use PositionUsedAs::*;

#[allow(clippy::unnecessary_lazy_evaluations)]
pub(crate) fn parse(
    s: &ast::String,
    fmt_snippet: Option<String>,
    mut args: FormatArgumentsCollector,
    is_direct_literal: bool,
    mut synth: impl FnMut(Name, Option<TextRange>) -> ExprId,
    mut record_usage: impl FnMut(Name, Option<TextRange>),
    call_ctx: SyntaxContext,
) -> FormatArgs {
    let Ok(text) = s.value() else {
        return FormatArgs {
            template: Default::default(),
            arguments: args.finish(),
            orphans: vec![],
        };
    };
    let str_style = match s.quote_offsets() {
        Some(offsets) => {
            let raw = usize::from(offsets.quotes.0.len()) - 1;
            // subtract 1 for the `r` prefix
            (raw != 0).then(|| raw - 1)
        }
        None => None,
    };
    let mut parser =
        parse::Parser::new(&text, str_style, fmt_snippet, false, parse::ParseMode::Format);

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
        // FIXME: Diagnose
        return FormatArgs {
            template: Default::default(),
            arguments: args.finish(),
            orphans: vec![],
        };
    }

    let to_span = |inner_span: std::ops::Range<usize>| {
        is_source_literal.then(|| {
            TextRange::new(inner_span.start.try_into().unwrap(), inner_span.end.try_into().unwrap())
        })
    };

    let mut used = vec![false; args.explicit_args().len()];
    let mut invalid_refs = Vec::new();
    let mut numeric_references_to_named_arg = Vec::new();

    enum ArgRef<'a> {
        Index(usize),
        Name(&'a str, Option<TextRange>),
    }
    let mut lookup_arg = |arg: ArgRef<'_>,
                          span: Option<TextRange>,
                          used_as: PositionUsedAs,
                          kind: FormatArgPositionKind|
     -> FormatArgPosition {
        let index = match arg {
            ArgRef::Index(index) => {
                if let Some(arg) = args.by_index(index) {
                    used[index] = true;
                    if arg.kind.ident().is_some() {
                        // This was a named argument, but it was used as a positional argument.
                        numeric_references_to_named_arg.push((index, span, used_as));
                    }
                    Ok(index)
                } else {
                    // Doesn't exist as an explicit argument.
                    invalid_refs.push((Either::Left(index), span, used_as, kind));
                    Err(Either::Left(index))
                }
            }
            ArgRef::Name(name, span) => {
                let name = Name::new(name, call_ctx);
                if let Some((index, _)) = args.by_name(&name) {
                    record_usage(name, span);
                    // Name found in `args`, so we resolve it to its index.
                    if index < args.explicit_args().len() {
                        // Mark it as used, if it was an explicit argument.
                        used[index] = true;
                    }
                    Ok(index)
                } else {
                    // Name not found in `args`, so we add it as an implicitly captured argument.
                    if !is_direct_literal {
                        // For the moment capturing variables from format strings expanded from macros is
                        // disabled (see RFC #2795)
                        // FIXME: Diagnose
                        invalid_refs.push((Either::Right(name.clone()), span, used_as, kind));
                        Err(Either::Right(name))
                    } else {
                        record_usage(name.clone(), span);
                        Ok(args.add(FormatArgument {
                            kind: FormatArgumentKind::Captured(name.clone()),
                            // FIXME: This is problematic, we might want to synthesize a dummy
                            // expression proper and/or desugar these.
                            expr: synth(name, span),
                        }))
                    }
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
            parse::Piece::Lit(s) => {
                unfinished_literal.push_str(s);
            }
            parse::Piece::NextArgument(arg) => {
                let parse::Argument { position, position_span, format } = *arg;
                if !unfinished_literal.is_empty() {
                    template.push(FormatArgsPiece::Literal(Symbol::intern(&unfinished_literal)));
                    unfinished_literal.clear();
                }

                let span =
                    parser.arg_places.get(placeholder_index).and_then(|s| to_span(s.clone()));
                placeholder_index += 1;

                let position_span = to_span(position_span);
                let argument = match position {
                    parse::ArgumentImplicitlyIs(i) => lookup_arg(
                        ArgRef::Index(i),
                        position_span,
                        Placeholder(span),
                        FormatArgPositionKind::Implicit,
                    ),
                    parse::ArgumentIs(i) => lookup_arg(
                        ArgRef::Index(i),
                        position_span,
                        Placeholder(span),
                        FormatArgPositionKind::Number,
                    ),
                    parse::ArgumentNamed(name) => lookup_arg(
                        ArgRef::Name(name, position_span),
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
                        // FIXME: Diagnose
                        FormatTrait::Display
                    }
                };

                let precision_span = format.precision_span.and_then(to_span);
                let precision = match format.precision {
                    parse::CountIs(n) => Some(FormatCount::Literal(n)),
                    parse::CountIsName(name, name_span) => Some(FormatCount::Argument(lookup_arg(
                        ArgRef::Name(name, to_span(name_span)),
                        precision_span,
                        Precision,
                        FormatArgPositionKind::Named,
                    ))),
                    parse::CountIsParam(i) => Some(FormatCount::Argument(lookup_arg(
                        ArgRef::Index(i),
                        precision_span,
                        Precision,
                        FormatArgPositionKind::Number,
                    ))),
                    parse::CountIsStar(i) => Some(FormatCount::Argument(lookup_arg(
                        ArgRef::Index(i),
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
                        ArgRef::Name(name, to_span(name_span)),
                        width_span,
                        Width,
                        FormatArgPositionKind::Named,
                    ))),
                    parse::CountIsParam(i) => Some(FormatCount::Argument(lookup_arg(
                        ArgRef::Index(i),
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
        // FIXME: Diagnose
    }

    let unused = used
        .iter()
        .enumerate()
        .filter(|&(_, used)| !used)
        .map(|(i, _)| {
            let named = matches!(args.explicit_args()[i].kind, FormatArgumentKind::Named(_));
            (args.explicit_args()[i].expr, named)
        })
        .collect::<Vec<_>>();

    if !unused.is_empty() {
        // FIXME: Diagnose
    }

    FormatArgs {
        template: template.into_boxed_slice(),
        arguments: args.finish(),
        orphans: unused.into_iter().map(TupleExt::head).collect(),
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FormatArgumentsCollector {
    arguments: Vec<FormatArgument>,
    num_unnamed_args: usize,
    num_explicit_args: usize,
    names: Vec<(Name, usize)>,
}

impl FormatArgumentsCollector {
    pub(crate) fn finish(self) -> FormatArguments {
        FormatArguments {
            arguments: self.arguments.into_boxed_slice(),
            num_unnamed_args: self.num_unnamed_args,
            num_explicit_args: self.num_explicit_args,
            names: self.names.into_boxed_slice(),
        }
    }

    pub fn add(&mut self, arg: FormatArgument) -> usize {
        let index = self.arguments.len();
        if let Some(name) = arg.kind.ident() {
            self.names.push((name.clone(), index));
        } else if self.names.is_empty() {
            // Only count the unnamed args before the first named arg.
            // (Any later ones are errors.)
            self.num_unnamed_args += 1;
        }
        if !matches!(arg.kind, FormatArgumentKind::Captured(..)) {
            // This is an explicit argument.
            // Make sure that all arguments so far are explicit.
            assert_eq!(
                self.num_explicit_args,
                self.arguments.len(),
                "captured arguments must be added last"
            );
            self.num_explicit_args += 1;
        }
        self.arguments.push(arg);
        index
    }

    pub fn by_name(&self, name: &Name) -> Option<(usize, &FormatArgument)> {
        let &(_, i) = self.names.iter().find(|(n, _)| n == name)?;
        Some((i, &self.arguments[i]))
    }

    pub fn by_index(&self, i: usize) -> Option<&FormatArgument> {
        (i < self.num_explicit_args).then(|| &self.arguments[i])
    }

    pub fn unnamed_args(&self) -> &[FormatArgument] {
        &self.arguments[..self.num_unnamed_args]
    }

    pub fn named_args(&self) -> &[FormatArgument] {
        &self.arguments[self.num_unnamed_args..self.num_explicit_args]
    }

    pub fn explicit_args(&self) -> &[FormatArgument] {
        &self.arguments[..self.num_explicit_args]
    }

    pub fn all_args(&self) -> &[FormatArgument] {
        &self.arguments[..]
    }

    pub fn all_args_mut(&mut self) -> &mut Vec<FormatArgument> {
        &mut self.arguments
    }
}

impl FormatArgumentKind {
    pub fn ident(&self) -> Option<&Name> {
        match self {
            Self::Normal => None,
            Self::Named(id) => Some(id),
            Self::Captured(id) => Some(id),
        }
    }
}
