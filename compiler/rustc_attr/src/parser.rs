use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::iter::Peekable;

use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{RefTokenTreeCursor, TokenTree};
use rustc_ast::{AttrArgs, DelimArgs, Expr, ExprKind, LitKind, MetaItemLit, NormalAttr, Path};
use rustc_ast_pretty::pprust;
use rustc_errors::DiagCtxtHandle;
use rustc_hir::{self as hir, AttrPath};
use rustc_span::symbol::{Ident, kw};
use rustc_span::{DUMMY_SP, Span, Symbol};

pub(crate) struct SegmentIterator<'a> {
    offset: usize,
    path: &'a PathParser<'a>,
}

impl<'a> Iterator for SegmentIterator<'a> {
    type Item = &'a Ident;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.path.len() {
            return None;
        }

        let res = match self.path {
            PathParser::Ast(ast_path) => &ast_path.segments[self.offset].ident,
            PathParser::Attr(attr_path) => &attr_path.segments[self.offset],
        };

        self.offset += 1;
        Some(res)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum PathParser<'a> {
    Ast(&'a Path),
    Attr(AttrPath),
}

impl<'a> PathParser<'a> {
    pub(crate) fn get_attribute_path(&self) -> hir::AttrPath {
        AttrPath {
            segments: self.segments().copied().collect::<Vec<_>>().into_boxed_slice(),
            span: self.span(),
        }
    }

    pub(crate) fn segments(&'a self) -> impl Iterator<Item = &'a Ident> {
        SegmentIterator { offset: 0, path: self }
    }

    pub(crate) fn span(&self) -> Span {
        match self {
            PathParser::Ast(path) => path.span,
            PathParser::Attr(attr_path) => attr_path.span,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            PathParser::Ast(path) => path.segments.len(),
            PathParser::Attr(attr_path) => attr_path.segments.len(),
        }
    }

    pub(crate) fn segments_is(&self, segments: &[Symbol]) -> bool {
        self.len() == segments.len() && self.segments().zip(segments).all(|(a, b)| a.name == *b)
    }

    pub(crate) fn word(&self) -> Option<Ident> {
        (self.len() == 1).then(|| **self.segments().next().as_ref().unwrap())
    }

    pub(crate) fn word_or_empty(&self) -> Ident {
        self.word().unwrap_or_else(Ident::empty)
    }

    /// Asserts that this MetaItem is some specific word.
    ///
    /// See [`word`](Self::word) for examples of what a word is.
    pub(crate) fn word_is(&self, sym: Symbol) -> bool {
        self.word().map(|i| i.name == sym).unwrap_or(false)
    }
}

impl Display for PathParser<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathParser::Ast(path) => write!(f, "{}", pprust::path_to_string(path)),
            PathParser::Attr(attr_path) => write!(f, "{attr_path}"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Args<'a, T: Clone> {
    Empty,
    Delimited(Cow<'a, DelimArgs>),
    Eq { eq_span: Span, value: Cow<'a, T>, value_span: Span },
}

impl<'a, T: Clone> Args<'a, T> {
    pub(crate) fn span(&self) -> Option<Span> {
        match self {
            Args::Empty => None,
            Args::Delimited(delim_args) => Some(delim_args.dspan.entire()),
            Args::Eq { eq_span, value_span, .. } => Some(value_span.with_lo(eq_span.lo())),
        }
    }
}

impl<'a> Args<'a, Expr> {
    fn from_attr_args(value: &'a AttrArgs) -> Self {
        match value {
            AttrArgs::Empty => Self::Empty,
            AttrArgs::Delimited(delim_args) => Self::Delimited(Cow::Borrowed(delim_args)),
            AttrArgs::Eq { eq_span, expr } => {
                Self::Eq { eq_span: *eq_span, value: Cow::Borrowed(expr), value_span: expr.span }
            }
        }
    }
}

impl Args<'_, MetaItemLit> {
    fn from_attr_args(value: AttrArgs, dcx: DiagCtxtHandle<'_>) -> Self {
        match value {
            AttrArgs::Empty => Self::Empty,
            AttrArgs::Delimited(delim_args) => Self::Delimited(Cow::Owned(delim_args)),
            AttrArgs::Eq { eq_span, expr } => Self::Eq {
                eq_span,
                value: Cow::Owned(expr_to_lit(dcx, &expr)),
                value_span: expr.span,
            },
        }
    }
}

#[must_use]
pub(crate) struct GenericArgParser<'a, T: Clone> {
    args: Args<'a, T>,
    dcx: DiagCtxtHandle<'a>,
}

pub(crate) trait ArgParser<'a> {
    type NameValueParser: NameValueParser<'a>;

    /// Asserts that this MetaItem is a list
    ///
    /// Some examples:
    ///
    /// - `#[allow(clippy::complexity)]`: `(clippy::complexity)` is a list
    /// - `#[rustfmt::skip::macros(target_macro_name)]`: `(target_macro_name)` is a list
    fn list<'s, 'r>(&'s self) -> Option<MetaItemListParser<'r>>
    where
        's: 'r,
        'a: 'r;

    /// Asserts that this MetaItem is a name-value pair.
    ///
    /// Some examples:
    ///
    /// - `#[clippy::cyclomatic_complexity = "100"]`: `clippy::cyclomatic_complexity = "100"` is a name value pair,
    ///   where the name is a path (`clippy::cyclomatic_complexity`). You already checked the path
    ///   to get an `ArgParser`, so this method will effectively only assert that the `= "100"` is
    ///   there
    /// - `#[doc = "hello"]`: `doc = "hello`  is also a name value pair
    fn name_value(&self) -> Option<Self::NameValueParser>;

    /// Asserts that there are no arguments
    fn no_args(&self) -> bool;
}

macro_rules! argparser {
    ($ty: ty) => {
        impl<'a> ArgParser<'a> for GenericArgParser<'a, $ty> {
            type NameValueParser = GenericNameValueParser<'a, $ty>;

            fn list<'s, 'r>(&'s self) -> Option<MetaItemListParser<'r>>
            where
                's: 'r,
                'a: 'r,
            {
                match &self.args {
                    Args::Delimited(args) if args.delim == Delimiter::Parenthesis => Some(
                        MetaItemListParserContext {
                            inside_delimiters: args.tokens.trees().peekable(),
                            dcx: self.dcx,
                        }
                        .parse()
                        .unwrap(),
                    ),
                    Args::Delimited(_) | Args::Eq { .. } | Args::Empty => None,
                }
            }

            fn name_value(&self) -> Option<GenericNameValueParser<'a, $ty>> {
                match &self.args {
                    Args::Eq { eq_span, value, value_span } => Some(GenericNameValueParser {
                        eq_span: *eq_span,
                        value: value.to_owned(),
                        value_span: *value_span,
                        dcx: self.dcx,
                    }),
                    Args::Delimited(_) | Args::Empty => None,
                }
            }

            fn no_args(&self) -> bool {
                matches!(&self.args, Args::Empty)
            }
        }
    };
}

argparser!(Expr);
argparser!(MetaItemLit);

/// Inside lists, values could be either literals, or more deeply nested meta items.
/// This enum represents that.
///
/// Choose which one you want using the provided methods.
#[derive(Debug)]
pub(crate) enum MetaItemOrLitParser<'a, T: Clone>
where
    GenericMetaItemParser<'a, T>: MetaItemParser<'a>,
{
    MetaItemParser(GenericMetaItemParser<'a, T>),
    Lit(MetaItemLit),
}

impl<'a, T: Clone> MetaItemOrLitParser<'a, T>
where
    GenericMetaItemParser<'a, T>: MetaItemParser<'a>,
{
    pub(crate) fn span(&self) -> Span {
        match self {
            MetaItemOrLitParser::MetaItemParser(generic_meta_item_parser) => {
                generic_meta_item_parser.span()
            }
            MetaItemOrLitParser::Lit(meta_item_lit) => meta_item_lit.span,
        }
    }

    pub(crate) fn lit(self) -> Option<MetaItemLit> {
        match self {
            MetaItemOrLitParser::Lit(meta_item_lit) => Some(meta_item_lit),
            _ => None,
        }
    }

    pub(crate) fn meta_item(self) -> Option<GenericMetaItemParser<'a, T>> {
        match self {
            MetaItemOrLitParser::MetaItemParser(parser) => Some(parser),
            _ => None,
        }
    }
}

/// Utility that deconstructs a MetaItem into usable parts.
///
/// MetaItems are syntactically extremely flexible, but specific attributes want to parse
/// them in custom, more restricted ways. This can be done using this struct.
///
/// MetaItems consist of some path, and some args. The args could be empty. In other words:
///
/// - `name` -> args are empty
/// - `name(...)` -> args are a [`list`](ArgParser::list), which is the bit between the parentheses
/// - `name = value`-> arg is [`name_value`](ArgParser::name_value), where the argument is the
///   `= value` part
///
/// The syntax of MetaItems can be found at <https://doc.rust-lang.org/reference/attributes.html>
pub(crate) struct GenericMetaItemParser<'a, T: Clone> {
    path: PathParser<'a>,
    args: Args<'a, T>,
    dcx: DiagCtxtHandle<'a>,
}

impl<'a, T: Clone + Debug> Debug for GenericMetaItemParser<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericMetaItemParser")
            .field("path", &self.path)
            .field("args", &self.args)
            .finish()
    }
}

impl<'a> GenericMetaItemParser<'a, Expr> {
    /// Create a new parser from a [`NormalAttr`], which is stored inside of any
    /// [`ast::Attribute`](Attribute)
    pub(crate) fn from_attr(attr: &'a NormalAttr, dcx: DiagCtxtHandle<'a>) -> Self {
        Self {
            path: PathParser::Ast(&attr.item.path),
            args: Args::<Expr>::from_attr_args(&attr.item.args),
            dcx,
        }
    }
}

pub(crate) trait MetaItemParser<'a>: Debug + 'a {
    type ArgParser: ArgParser<'a>;

    fn span(&self) -> Span;

    /// Gets just the path, without the args.
    fn path_without_args(&self) -> PathParser<'a>;

    /// Gets just the args parser, without caring about the path.
    fn args(&self) -> Self::ArgParser;

    fn deconstruct(&self) -> (PathParser<'a>, Self::ArgParser) {
        (self.path_without_args(), self.args())
    }

    /// Asserts that this MetaItem starts with a path. Some examples:
    ///
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path
    /// - `#[allow(clippy::complexity)]`: `clippy::complexity` is a path
    /// - `#[inline]`: `inline` is a single segment path
    /// - `#[inline(always)]`: `always` is a single segment path, but `inline` is *not and
    ///    should be parsed using [`list`](Self::list)
    fn path(&self) -> (PathParser<'a>, Self::ArgParser) {
        self.deconstruct()
    }

    /// Asserts that this MetaItem starts with a word, or single segment path.
    /// Doesn't return the args parser.
    ///
    /// For examples. see [`Self::word`]
    fn word_without_args(&self) -> Option<Ident> {
        Some(self.word()?.0)
    }

    /// Like [`word`](Self::word), but returns an empty symbol instead of None
    fn word_or_empty_without_args(&self) -> Ident {
        self.word_or_empty().0
    }

    /// Asserts that this MetaItem starts with a word, or single segment path.
    ///
    /// Some examples:
    /// - `#[inline]`: `inline` is a word
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path, and not a word
    /// - `#[inline(always)]`: `always` is a word, but `inline` is *not and should be parsed
    ///   using [`path_list`](Self::path_list)
    /// - `#[allow(clippy::complexity)]`: `clippy::complexity` is *not* a word, and should instead be parsed
    ///   using [`path`](Self::path)
    fn word(&self) -> Option<(Ident, Self::ArgParser)> {
        let (path, args) = self.deconstruct();
        Some((path.word()?, args))
    }

    /// Like [`word`](Self::word), but returns an empty symbol instead of None
    fn word_or_empty(&self) -> (Ident, Self::ArgParser) {
        let (path, args) = self.deconstruct();
        (path.word().unwrap_or(Ident::empty()), args)
    }

    /// Asserts that this MetaItem starts with some specific word.
    ///
    /// See [`word`](Self::word) for examples of what a word is.
    fn word_is(&self, sym: Symbol) -> Option<Self::ArgParser> {
        self.path_without_args().word_is(sym).then(|| self.args())
    }

    /// Asserts that this MetaItem starts with some specific path.
    ///
    /// See [`word`](Self::path) for examples of what a word is.
    fn path_is(&self, segments: &[Symbol]) -> Option<Self::ArgParser> {
        self.path_without_args().segments_is(segments).then(|| self.args())
    }
}

impl<'a> MetaItemParser<'a> for GenericMetaItemParser<'a, Expr> {
    type ArgParser = GenericArgParser<'a, Expr>;

    fn span(&self) -> Span {
        if let Some(other) = self.args.span() {
            self.path.span().with_hi(other.hi())
        } else {
            self.path.span()
        }
    }

    /// Gets just the path, without the args.
    fn path_without_args(&self) -> PathParser<'a> {
        self.path.clone()
    }

    /// Gets just the args parser, without caring about the path.
    fn args(&self) -> GenericArgParser<'a, Expr> {
        GenericArgParser { args: self.args.clone(), dcx: self.dcx }
    }
}

impl<'a> MetaItemParser<'a> for GenericMetaItemParser<'a, MetaItemLit> {
    type ArgParser = GenericArgParser<'a, MetaItemLit>;

    fn span(&self) -> Span {
        if let Some(other) = self.args.span() {
            self.path.span().with_hi(other.hi())
        } else {
            self.path.span()
        }
    }

    /// Gets just the path, without the args.
    fn path_without_args(&self) -> PathParser<'a> {
        self.path.clone()
    }

    /// Gets just the args parser, without caring about the path.
    fn args(&self) -> GenericArgParser<'a, MetaItemLit> {
        GenericArgParser { args: self.args.clone(), dcx: self.dcx }
    }
}

pub(crate) struct GenericNameValueParser<'a, T: Clone> {
    pub(crate) eq_span: Span,
    value: Cow<'a, T>,
    pub(crate) value_span: Span,
    dcx: DiagCtxtHandle<'a>,
}

impl<'a, T: Clone + Debug> Debug for GenericNameValueParser<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericNameValueParser")
            .field("eq_span", &self.eq_span)
            .field("value", &self.value)
            .field("value_span", &self.value_span)
            .finish()
    }
}

pub(crate) trait NameValueParser<'a>: Debug {
    fn value_span(&self) -> Span;

    fn value_as_lit(&self) -> MetaItemLit;
    fn value_as_str(&self) -> Option<Symbol> {
        self.value_as_lit().kind.str()
    }
}

impl<'a> NameValueParser<'a> for GenericNameValueParser<'a, Expr> {
    fn value_as_lit(&self) -> MetaItemLit {
        expr_to_lit(self.dcx, &self.value)
    }

    fn value_span(&self) -> Span {
        self.value_span
    }
}

impl<'a> NameValueParser<'a> for GenericNameValueParser<'a, MetaItemLit> {
    fn value_as_lit(&self) -> MetaItemLit {
        self.value.clone().into_owned()
    }
    fn value_span(&self) -> Span {
        self.value_span
    }
}

fn expr_to_lit(dcx: DiagCtxtHandle<'_>, expr: &Expr) -> MetaItemLit {
    // In valid code the value always ends up as a single literal. Otherwise, a dummy
    // literal suffices because the error is handled elsewhere.
    if let ExprKind::Lit(token_lit) = expr.kind
        && let Ok(lit) = MetaItemLit::from_token_lit(token_lit, expr.span)
    {
        lit
    } else {
        let guar = dcx.has_errors().unwrap();
        MetaItemLit { symbol: kw::Empty, suffix: None, kind: LitKind::Err(guar), span: DUMMY_SP }
    }
}

struct MetaItemListParserContext<'a> {
    // the tokens inside the delimiters, so `#[some::attr(a b c)]` would have `a b c` inside
    inside_delimiters: Peekable<RefTokenTreeCursor<'a>>,
    dcx: DiagCtxtHandle<'a>,
}

impl<'a> MetaItemListParserContext<'a> {
    fn done(&mut self) -> bool {
        self.inside_delimiters.peek().is_none()
    }

    fn next_path(&mut self) -> Option<AttrPath> {
        // FIXME: Share code with `parse_path`.
        let tt = self.inside_delimiters.next().map(|tt| TokenTree::uninterpolate(tt));

        match tt.as_deref() {
            Some(&TokenTree::Token(
                Token { kind: ref kind @ (token::Ident(..) | token::PathSep), span },
                _,
            )) => {
                let mut segments = if let &token::Ident(name, _) = kind {
                    if let Some(TokenTree::Token(Token { kind: token::PathSep, .. }, _)) =
                        self.inside_delimiters.peek()
                    {
                        self.inside_delimiters.next();
                        vec![Ident::new(name, span)]
                    } else {
                        return Some(AttrPath {
                            segments: vec![Ident::new(name, span)].into_boxed_slice(),
                            span,
                        });
                    }
                } else {
                    vec![Ident::new(kw::PathRoot, span)]
                };
                loop {
                    if let Some(&TokenTree::Token(Token { kind: token::Ident(name, _), span }, _)) =
                        self.inside_delimiters
                            .next()
                            .map(|tt| TokenTree::uninterpolate(tt))
                            .as_deref()
                    {
                        segments.push(Ident::new(name, span));
                    } else {
                        return None;
                    }
                    if let Some(TokenTree::Token(Token { kind: token::PathSep, .. }, _)) =
                        self.inside_delimiters.peek()
                    {
                        self.inside_delimiters.next();
                    } else {
                        break;
                    }
                }
                let span = span.with_hi(segments.last().unwrap().span.hi());
                Some(AttrPath { segments: segments.into_boxed_slice(), span })
            }
            Some(TokenTree::Token(
                Token { kind: token::OpenDelim(_) | token::CloseDelim(_), .. },
                _,
            )) => {
                panic!("Should be `AttrTokenTree::Delimited`, not delim tokens: {:?}", tt);
            }
            _ => return None,
        }
    }

    fn value(&mut self) -> Option<MetaItemLit> {
        match self.inside_delimiters.next() {
            Some(TokenTree::Delimited(.., Delimiter::Invisible(_), inner_tokens)) => {
                MetaItemListParserContext {
                    inside_delimiters: inner_tokens.trees().peekable(),
                    dcx: self.dcx,
                }
                .value()
            }
            Some(TokenTree::Token(ref token, _)) => MetaItemLit::from_token(token),
            _ => None,
        }
    }

    fn next(&mut self) -> Option<MetaItemOrLitParser<'a, MetaItemLit>> {
        // a list element is either a literal
        if let Some(TokenTree::Token(token, _)) = self.inside_delimiters.peek()
            && let Some(lit) = MetaItemLit::from_token(token)
        {
            self.inside_delimiters.next();
            return Some(MetaItemOrLitParser::Lit(lit));
        }

        // or a path.
        let path = self.next_path()?;

        // Paths can be followed by:
        // - `(more meta items)` (another list)
        // - `= lit` (a name-value)
        // - nothing
        Some(MetaItemOrLitParser::MetaItemParser(match self.inside_delimiters.peek() {
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. }, _)) => match &**nt {
                token::Nonterminal::NtMeta(item) => GenericMetaItemParser {
                    path: PathParser::Ast(&item.path),
                    args: Args::<MetaItemLit>::from_attr_args(item.args.clone(), self.dcx),
                    dcx: self.dcx,
                },
                token::Nonterminal::NtPath(path) => GenericMetaItemParser {
                    path: PathParser::Ast(path),
                    args: Args::Empty,
                    dcx: self.dcx,
                },
                _ => return None,
            },
            Some(TokenTree::Delimited(dspan, _, delim @ Delimiter::Parenthesis, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                self.inside_delimiters.next();

                GenericMetaItemParser {
                    path: PathParser::Attr(path),
                    args: Args::Delimited(Cow::Owned(DelimArgs {
                        dspan: *dspan,
                        delim: *delim,
                        tokens: inner_tokens,
                    })),
                    dcx: self.dcx,
                }
            }
            // FIXME(jdonszelmann) nice error?
            // tests seem to say its already parsed and rejected maybe?
            Some(TokenTree::Delimited(..)) => return None,
            Some(TokenTree::Token(Token { kind: token::Eq, span }, _)) => {
                self.inside_delimiters.next();
                let value = self.value()?;
                GenericMetaItemParser {
                    path: PathParser::Attr(path),
                    args: Args::Eq {
                        eq_span: *span,
                        value_span: value.span,
                        value: Cow::Owned(value),
                    },
                    dcx: self.dcx,
                }
            }
            _ => GenericMetaItemParser {
                path: PathParser::Attr(path),
                args: Args::Empty,
                dcx: self.dcx,
            },
        }))
    }

    pub(crate) fn parse(mut self) -> Option<MetaItemListParser<'a>> {
        let mut sub_parsers = Vec::new();

        while !self.done() {
            sub_parsers.push(self.next()?);
            match self.inside_delimiters.next() {
                None | Some(TokenTree::Token(Token { kind: token::Comma, .. }, _)) => {}
                _ => return None,
            }
        }

        Some(MetaItemListParser { sub_parsers })
    }
}

#[derive(Debug)]
pub(crate) struct MetaItemListParser<'a> {
    sub_parsers: Vec<MetaItemOrLitParser<'a, MetaItemLit>>,
}

impl<'a> MetaItemListParser<'a> {
    /// Lets you pick and choose as what you want to parse each element in the list
    pub(crate) fn mixed(self) -> impl Iterator<Item = MetaItemOrLitParser<'a, MetaItemLit>> + 'a {
        self.sub_parsers.into_iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.sub_parsers.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Asserts that every item in the list is another list starting with a word.
    ///
    /// See [`MetaItemParser::word`] for examples of words.
    pub(crate) fn all_word_list(self) -> Option<Vec<(Ident, GenericArgParser<'a, MetaItemLit>)>> {
        self.mixed().map(|i| i.meta_item()?.word()).collect()
    }

    /// Asserts that every item in the list is another list starting with a full path.
    ///
    /// See [`MetaItemParser::path`] for examples of paths.
    pub(crate) fn all_path_list(
        self,
    ) -> Option<Vec<(PathParser<'a>, GenericArgParser<'a, MetaItemLit>)>> {
        self.mixed().map(|i| Some(i.meta_item()?.path())).collect()
    }

    /// Returns Some if the list contains only a single element.
    ///
    /// Inside the Some is the parser to parse this single element.
    pub(crate) fn single(self) -> Option<MetaItemOrLitParser<'a, MetaItemLit>> {
        let mut iter = self.mixed();
        iter.next().filter(|_| iter.next().is_none())
    }
}
