use rustc_ast::ptr::P;
use rustc_ast::Expr;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

// Definitions:
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
// └──────────────────────────────────────────────┘
//                     FormatArgs
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//                                     └─────────┘
//                                      argument
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//              └───────────────────┘
//                     template
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//               └────┘└─────────┘└┘
//                      pieces
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//               └────┘           └┘
//                   literal pieces
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//                     └─────────┘
//                     placeholder
//
// format_args!("hello {abc:.xyz$}!!", abc="world");
//                      └─┘  └─┘
//                      positions (could be names, numbers, empty, or `*`)

/// (Parsed) format args.
///
/// Basically the "AST" for a complete `format_args!()`.
///
/// E.g., `format_args!("hello {name}");`.
#[derive(Clone, Debug)]
pub struct FormatArgs {
    pub span: Span,
    pub template: Vec<FormatArgsPiece>,
    pub arguments: Vec<(P<Expr>, FormatArgKind)>,
}

#[derive(Clone, Debug)]
pub enum FormatArgsPiece {
    Literal(Symbol),
    Placeholder(FormatPlaceholder),
}

#[derive(Clone, Debug)]
pub enum FormatArgKind {
    /// `format_args(…, arg)`
    Normal,
    /// `format_args(…, arg = 1)`
    Named(Ident),
    /// `format_args("… {arg} …")`
    Captured(Ident),
}

impl FormatArgKind {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            &Self::Normal => None,
            &Self::Named(id) => Some(id),
            &Self::Captured(id) => Some(id),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatPlaceholder {
    /// Index into [`FormatArgs::arguments`].
    pub argument: FormatArgPosition,
    /// The span inside the format string for the full `{…}` placeholder.
    pub span: Option<Span>,
    /// `{}`, `{:?}`, or `{:x}`, etc.
    pub format_trait: FormatTrait,
    /// `{}` or `{:.5}` or `{:-^20}`, etc.
    pub format_options: FormatOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatArgPosition {
    /// Which argument this position refers to (Ok),
    /// or would've referred to if it existed (Err).
    pub index: Result<usize, usize>,
    /// What kind of position this is. See [`FormatArgPositionKind`].
    pub kind: FormatArgPositionKind,
    /// The span of the name or number.
    pub span: Option<Span>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FormatOptions {
    /// The width. E.g. `{:5}` or `{:width$}`.
    pub width: Option<FormatCount>,
    /// The precision. E.g. `{:.5}` or `{:.precision$}`.
    pub precision: Option<FormatCount>,
    /// The alignment. E.g. `{:>}` or `{:<}` or `{:^}`.
    pub alignment: Option<FormatAlignment>,
    /// The fill character. E.g. the `.` in `{:.>10}`.
    pub fill: Option<char>,
    /// The `+`, `-`, `0`, `#`, `x?` and `X?` flags.
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// `{:0}` or `{:.0}`
    Literal(usize),
    /// `{:.*}`, `{:.0$}`, or `{:a$}`, etc.
    Argument(FormatArgPosition),
}
