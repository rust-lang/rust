use crate::ptr::P;
use crate::Expr;
use rustc_data_structures::fx::FxHashMap;
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
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArgs {
    pub span: Span,
    pub template: Vec<FormatArgsPiece>,
    pub arguments: FormatArguments,
}

/// A piece of a format template string.
///
/// E.g. "hello" or "{name}".
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FormatArgsPiece {
    Literal(Symbol),
    Placeholder(FormatPlaceholder),
}

/// The arguments to format_args!().
///
/// E.g. `1, 2, name="ferris", n=3`,
/// but also implicit captured arguments like `x` in `format_args!("{x}")`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArguments {
    arguments: Vec<FormatArgument>,
    num_unnamed_args: usize,
    num_explicit_args: usize,
    names: FxHashMap<Symbol, usize>,
}

// FIXME: Rustdoc has trouble proving Send/Sync for this. See #106930.
#[cfg(parallel_compiler)]
unsafe impl Sync for FormatArguments {}
#[cfg(parallel_compiler)]
unsafe impl Send for FormatArguments {}

impl FormatArguments {
    pub fn new() -> Self {
        Self {
            arguments: Vec::new(),
            names: FxHashMap::default(),
            num_unnamed_args: 0,
            num_explicit_args: 0,
        }
    }

    pub fn add(&mut self, arg: FormatArgument) -> usize {
        let index = self.arguments.len();
        if let Some(name) = arg.kind.ident() {
            self.names.insert(name.name, index);
        } else if self.names.is_empty() {
            // Only count the unnamed args before the first named arg.
            // (Any later ones are errors.)
            self.num_unnamed_args += 1;
        }
        if !matches!(arg.kind, FormatArgumentKind::Captured(..)) {
            // This is an explicit argument.
            // Make sure that all arguments so far are explcit.
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

    pub fn by_name(&self, name: Symbol) -> Option<(usize, &FormatArgument)> {
        let i = *self.names.get(&name)?;
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

    pub fn all_args_mut(&mut self) -> &mut [FormatArgument] {
        &mut self.arguments[..]
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArgument {
    pub kind: FormatArgumentKind,
    pub expr: P<Expr>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FormatArgumentKind {
    /// `format_args(…, arg)`
    Normal,
    /// `format_args(…, arg = 1)`
    Named(Ident),
    /// `format_args("… {arg} …")`
    Captured(Ident),
}

impl FormatArgumentKind {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            &Self::Normal => None,
            &Self::Named(id) => Some(id),
            &Self::Captured(id) => Some(id),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
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

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub struct FormatArgPosition {
    /// Which argument this position refers to (Ok),
    /// or would've referred to if it existed (Err).
    pub index: Result<usize, usize>,
    /// What kind of position this is. See [`FormatArgPositionKind`].
    pub kind: FormatArgPositionKind,
    /// The span of the name or number.
    pub span: Option<Span>,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatArgPositionKind {
    /// `{}` or `{:.*}`
    Implicit,
    /// `{1}` or `{:1$}` or `{:.1$}`
    Number,
    /// `{a}` or `{:a$}` or `{:.a$}`
    Named,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Encodable, Decodable, Default, Debug, PartialEq, Eq)]
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

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatSign {
    /// The `+` flag.
    Plus,
    /// The `-` flag.
    Minus,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatDebugHex {
    /// The `x` flag in `{:x?}`.
    Lower,
    /// The `X` flag in `{:X?}`.
    Upper,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatAlignment {
    /// `{:<}`
    Left,
    /// `{:>}`
    Right,
    /// `{:^}`
    Center,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatCount {
    /// `{:5}` or `{:.5}`
    Literal(usize),
    /// `{:.*}`, `{:.5$}`, or `{:a$}`, etc.
    Argument(FormatArgPosition),
}
