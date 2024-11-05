use std::borrow::Cow;
use std::fmt::Debug;
use std::iter::Peekable;

use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{RefTokenTreeCursor, TokenTree};
use rustc_ast::{AttrArgs, DelimArgs, Expr, ExprKind, LitKind, MetaItemLit, NormalAttr, Path};
use rustc_errors::DiagCtxtHandle;
use rustc_hir::{self as hir, AttrPath};
use rustc_span::symbol::{Ident, kw};
use rustc_span::{DUMMY_SP, Span, Symbol};

pub struct SegmentIterator<'a> {
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

#[derive(Clone)]
pub enum PathParser<'a> {
    Ast(&'a Path),
    Attr(AttrPath),
}

impl<'a> PathParser<'a> {
    pub fn get_attribute_path(&self) -> hir::AttrPath {
        AttrPath {
            segments: self.segments().copied().collect::<Vec<_>>().into_boxed_slice(),
            span: self.span(),
        }
    }

    pub fn segments(&'a self) -> impl Iterator<Item = &'a Ident> {
        SegmentIterator { offset: 0, path: self }
    }

    pub fn span(&self) -> Span {
        match self {
            PathParser::Ast(path) => path.span,
            PathParser::Attr(attr_path) => attr_path.span,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PathParser::Ast(path) => path.segments.len(),
            PathParser::Attr(attr_path) => attr_path.segments.len(),
        }
    }

    pub fn segments_is(&self, segments: &[Symbol]) -> bool {
        self.len() == segments.len() && self.segments().zip(segments).all(|(a, b)| a.name == *b)
    }

    pub fn word(&self) -> Option<Ident> {
        (self.len() == 1).then(|| **self.segments().next().as_ref().unwrap())
    }

    /// Asserts that this MetaItem is some specific word.
    ///
    /// See [`word`](Self::word) for examples of what a word is.
    pub fn word_is(&self, sym: Symbol) -> bool {
        self.word().map(|i| i.name == sym).unwrap_or(false)
    }
}

#[derive(Clone, Debug)]
pub enum Args<T> {
    Empty,
    Delimited(DelimArgs),
    Eq { eq_span: Span, value: T },
}

impl Args<Expr> {
    fn from_attr_args(value: AttrArgs) -> Self {
        match value {
            AttrArgs::Empty => Self::Empty,
            AttrArgs::Delimited(delim_args) => Self::Delimited(delim_args),
            AttrArgs::Eq { eq_span, expr } => Self::Eq { eq_span, value: expr.into_inner() },
        }
    }
}

impl Args<MetaItemLit> {
    fn from_attr_args(value: AttrArgs, dcx: DiagCtxtHandle<'_>) -> Self {
        match value {
            AttrArgs::Empty => Self::Empty,
            AttrArgs::Delimited(delim_args) => Self::Delimited(delim_args),
            AttrArgs::Eq { eq_span, expr } => {
                Self::Eq { eq_span, value: expr_to_lit(dcx, &expr) }
            }
        }
    }
}

/// Utility that deconstructs a MetaItem into usable parts.
///
/// MetaItems are syntactically extremely flexible, but specific attributes want to parse
/// them in custom, more restricted ways. This can be done using this struct.
///
/// The syntax of MetaItems can be found at <https://doc.rust-lang.org/reference/attributes.html>
pub struct MetaItemParser<'a, T> {
    path: PathParser<'a>,
    args: Args<T>,
    dcx: DiagCtxtHandle<'a>,
}

impl<'a> MetaItemParser<'a, Expr> {
    /// Create a new parser from a [`NormalAttr`], which is stored inside of any
    /// [`ast::Attribute`](Attribute)
    pub fn from_attr(attr: &'a NormalAttr, dcx: DiagCtxtHandle<'a>) -> Self {
        Self {
            path: PathParser::Ast(&attr.item.path),
            args: Args::<Expr>::from_attr_args(attr.item.args.clone()),
            dcx,
        }
    }
}

impl<'a, T: Clone> MetaItemParser<'a, T> {
    fn get_path_parser(&self) -> PathParser<'a> {
        self.path.clone()
    }

    /// Asserts that this MetaItem is a path. Some examples:
    ///
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path
    /// - `#[allow(clippy::complexity)]`: `clippy::complexity` is a path
    /// - `#[inline]`: `inline` is a single segment path
    /// - `#[inline(always)]`: `always` is a single segment path, but `inline` is *not and
    ///    should be parsed using [`list`](Self::list)
    pub fn path(&self) -> Option<PathParser<'a>> {
        self.empty_args().then(|| self.path.clone())
    }

    /// Asserts that this MetaItem is a word, or single segment path.
    ///
    /// Some examples:
    /// - `#[inline]`: `inline` is a word
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path, and not a word
    /// - `#[inline(always)]`: `always` is a word, but `inline` is *not and should be parsed
    ///   using [`path_list`](Self::path_list)
    /// - `#[allow(clippy::complexity)]`: `clippy::complexity` is *not* a word, and should instead be parsed
    ///   using [`path`](Self::path)
    pub fn word(&self) -> Option<Ident> {
        self.empty_args().then(|| self.get_path_parser().word()).flatten()
    }

    fn empty_args(&self) -> bool {
        matches!(self.args, Args::Empty)
    }

    /// Asserts that this MetaItem is some specific word.
    ///
    /// See [`word`](Self::word) for examples of what a word is.
    pub fn word_is(&self, sym: Symbol) -> bool {
        self.word().map(|i| i.name == sym).unwrap_or(false)
    }

    /// Asserts that this MetaItem is a list, starting with a path.
    ///
    /// Some examples:
    ///
    /// - `#[rustfmt::skip::macros(target_macro_name)]`: `rustfmt::skip::macros` is a path
    /// - `#[allow(clippy::complexity)]`: `allow` is a single segment path
    pub fn path_list(&'a self) -> Option<(PathParser<'a>, MetaItemListParser<'a>)> {
        Some((self.get_path_parser(), self.list()?))
    }

    /// Asserts that this MetaItem is a list, starting with a word.
    ///
    /// Some examples:
    ///
    /// - `#[allow(clippy::complexity)]`: `allow` is a word
    /// - `#[rustfmt::skip::macros(target_macro_name)]`: `rustfmt::skip::macros` is a path, so you'd
    ///   need [`path_list`](Self::path_list) to parse this.
    pub fn word_list(&'a self) -> Option<(Ident, MetaItemListParser<'a>)> {
        Some((self.get_path_parser().word()?, self.list()?))
    }

    /// Like [`path_list`](Self::path_list) but always compares the segments first
    pub fn path_list_is(&'a self, segments: &[Symbol]) -> Option<MetaItemListParser<'a>> {
        self.get_path_parser().segments_is(segments).then(|| self.list()).flatten()
    }

    /// Like [`word_list`](Self::word_list) but always compares the word first
    pub fn word_list_is(&'a self, word: Symbol) -> Option<MetaItemListParser<'a>> {
        self.get_path_parser().word_is(word).then(|| self.list()).flatten()
    }

    fn list(&'a self) -> Option<MetaItemListParser<'a>> {
        match &self.args {
            Args::Delimited(args) if args.delim == Delimiter::Parenthesis => {
                MetaItemListParserContext {
                    inside_delimiters: args.tokens.trees().peekable(),
                    dcx: self.dcx,
                }
                .parse()
            }
            Args::Delimited(_) | Args::Eq { .. } | Args::Empty => None,
        }
    }

    fn name_value(&'a self) -> Option<EqParser<'a, T>> {
        match &self.args {
            Args::Eq { eq_span, value } => {
                Some(EqParser { eq_span: *eq_span, value: Cow::Borrowed(value), dcx: self.dcx })
            }
            Args::Delimited(_) | Args::Empty => None,
        }
    }

    /// Asserts that this MetaItem is a name-value pair, starting with a path.
    ///
    /// Some examples:
    ///
    /// - `#[clippy::cyclomatic_complexity = "100"]`: `clippy::cyclomatic_complexity` is a path in a
    ///   name-value attribute
    /// - `#[doc = "hello"]`: `doc` is a single segment path in a name-value attribute
    pub fn path_name_value(&'a self) -> Option<(PathParser<'a>, EqParser<'a, T>)> {
        Some((self.get_path_parser(), self.name_value()?))
    }

    /// Asserts that this MetaItem is a name-value pair, starting with a path.
    ///
    /// Some examples:
    ///
    /// - `#[doc = "hello"]`: `doc` is a word in a name-value attribute
    /// - `#[clippy::cyclomatic_complexity = "100"]`: `clippy::cyclomatic_complexity`, so you'd
    ///   need [`path_name_value`](Self::path_name_value) to parse this.
    pub fn word_name_value(&'a self) -> Option<(Ident, EqParser<'a, T>)> {
        Some((self.get_path_parser().word()?, self.name_value()?))
    }

    /// Like [`path_name_value`](Self::path_name_value) but always compares the segments first
    pub fn path_name_value_is(&'a self, segments: &[Symbol]) -> Option<EqParser<'a, T>> {
        self.get_path_parser().segments_is(segments).then(|| self.name_value()).flatten()
    }

    /// Like [`word_name_value`](Self::word_name_value) but always compares the word first
    pub fn word_name_value_is(&'a self, word: Symbol) -> Option<EqParser<'a, T>> {
        self.get_path_parser().word_is(word).then(|| self.name_value()).flatten()
    }
}

pub struct EqParser<'a, T: Clone> {
    pub eq_span: Span,
    value: Cow<'a, T>,
    dcx: DiagCtxtHandle<'a>,
}

impl<'a> EqParser<'a, Expr> {
    /// Returns the value as an ast expression.
    pub fn value_as_ast_expr(&'a self) -> &'a Expr {
        &self.value
    }

    pub fn value_as_lit(&self) -> MetaItemLit {
        expr_to_lit(self.dcx, &self.value)
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

impl<'a> EqParser<'a, MetaItemLit> {
    pub fn value_as_lit(&self) -> MetaItemLit {
        self.value.clone().into_owned()
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

    fn next(&mut self) -> Option<MetaItemParser<'a, MetaItemLit>> {
        let path = self.next_path()?;

        Some(match self.inside_delimiters.peek() {
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. }, _)) => match &**nt {
                token::Nonterminal::NtMeta(item) => MetaItemParser {
                    path: PathParser::Ast(&item.path),
                    args: Args::<MetaItemLit>::from_attr_args(item.args.clone(), self.dcx),
                    dcx: self.dcx,
                },
                token::Nonterminal::NtPath(path) => {
                    MetaItemParser { path: PathParser::Ast(path), args: Args::Empty, dcx: self.dcx }
                }
                _ => return None,
            },
            Some(TokenTree::Delimited(dspan, _, delim @ Delimiter::Parenthesis, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                self.inside_delimiters.next();

                MetaItemParser {
                    path: PathParser::Attr(path),
                    args: Args::Delimited(DelimArgs {
                        dspan: *dspan,
                        delim: *delim,
                        tokens: inner_tokens,
                    }),
                    dcx: self.dcx,
                }
            }
            // FIXME(jdonszelmann) nice error?
            // tests seem to say its already parsed and rejected maybe?
            Some(TokenTree::Delimited(..)) => return None,
            Some(TokenTree::Token(Token { kind: token::Eq, span }, _)) => {
                self.inside_delimiters.next();
                MetaItemParser {
                    path: PathParser::Attr(path),
                    args: Args::Eq { eq_span: *span, value: self.value()? },
                    dcx: self.dcx,
                }
            }
            _ => MetaItemParser { path: PathParser::Attr(path), args: Args::Empty, dcx: self.dcx },
        })
    }

    pub fn parse(mut self) -> Option<MetaItemListParser<'a>> {
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

pub struct MetaItemListParser<'a> {
    sub_parsers: Vec<MetaItemParser<'a, MetaItemLit>>,
}
