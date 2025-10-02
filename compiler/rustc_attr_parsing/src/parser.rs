//! This is in essence an (improved) duplicate of `rustc_ast/attr/mod.rs`.
//! That module is intended to be deleted in its entirety.
//!
//! FIXME(jdonszelmann): delete `rustc_ast/attr/mod.rs`

use std::borrow::Cow;
use std::fmt::{Debug, Display};

use rustc_ast::token::{self, Delimiter, MetaVarKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrArgs, DelimArgs, Expr, ExprKind, LitKind, MetaItemLit, NormalAttr, Path};
use rustc_ast_pretty::pprust;
use rustc_errors::{Diag, PResult};
use rustc_hir::{self as hir, AttrPath};
use rustc_parse::exp;
use rustc_parse::parser::{Parser, PathStyle, token_descr};
use rustc_session::errors::{create_lit_error, report_lit_error};
use rustc_session::parse::ParseSess;
use rustc_span::{ErrorGuaranteed, Ident, Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::ShouldEmit;
use crate::session_diagnostics::{
    InvalidMetaItem, InvalidMetaItemQuoteIdentSugg, InvalidMetaItemRemoveNegSugg, MetaBadDelim,
    MetaBadDelimSugg, SuffixedLiteralInAttribute,
};

#[derive(Clone, Debug)]
pub struct PathParser<'a>(pub Cow<'a, Path>);

impl<'a> PathParser<'a> {
    pub fn get_attribute_path(&self) -> hir::AttrPath {
        AttrPath {
            segments: self.segments().copied().collect::<Vec<_>>().into_boxed_slice(),
            span: self.span(),
        }
    }

    pub fn segments(&'a self) -> impl Iterator<Item = &'a Ident> {
        self.0.segments.iter().map(|seg| &seg.ident)
    }

    pub fn span(&self) -> Span {
        self.0.span
    }

    pub fn len(&self) -> usize {
        self.0.segments.len()
    }

    pub fn segments_is(&self, segments: &[Symbol]) -> bool {
        self.segments().map(|segment| &segment.name).eq(segments)
    }

    pub fn word(&self) -> Option<Ident> {
        (self.len() == 1).then(|| **self.segments().next().as_ref().unwrap())
    }

    pub fn word_sym(&self) -> Option<Symbol> {
        self.word().map(|ident| ident.name)
    }

    /// Asserts that this MetaItem is some specific word.
    ///
    /// See [`word`](Self::word) for examples of what a word is.
    pub fn word_is(&self, sym: Symbol) -> bool {
        self.word().map(|i| i.name == sym).unwrap_or(false)
    }

    /// Checks whether the first segments match the givens.
    ///
    /// Unlike [`segments_is`](Self::segments_is),
    /// `self` may contain more segments than the number matched  against.
    pub fn starts_with(&self, segments: &[Symbol]) -> bool {
        segments.len() < self.len() && self.segments().zip(segments).all(|(a, b)| a.name == *b)
    }
}

impl Display for PathParser<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", pprust::path_to_string(&self.0))
    }
}

#[derive(Clone, Debug)]
#[must_use]
pub enum ArgParser<'a> {
    NoArgs,
    List(MetaItemListParser<'a>),
    NameValue(NameValueParser),
}

impl<'a> ArgParser<'a> {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::NoArgs => None,
            Self::List(l) => Some(l.span),
            Self::NameValue(n) => Some(n.value_span.with_lo(n.eq_span.lo())),
        }
    }

    pub fn from_attr_args<'sess>(
        value: &'a AttrArgs,
        parts: &[Symbol],
        psess: &'sess ParseSess,
        should_emit: ShouldEmit,
    ) -> Option<Self> {
        Some(match value {
            AttrArgs::Empty => Self::NoArgs,
            AttrArgs::Delimited(args) => {
                // The arguments of rustc_dummy are not validated if the arguments are delimited
                if parts == &[sym::rustc_dummy] {
                    return Some(ArgParser::List(MetaItemListParser {
                        sub_parsers: ThinVec::new(),
                        span: args.dspan.entire(),
                    }));
                }

                if args.delim != Delimiter::Parenthesis {
                    psess.dcx().emit_err(MetaBadDelim {
                        span: args.dspan.entire(),
                        sugg: MetaBadDelimSugg { open: args.dspan.open, close: args.dspan.close },
                    });
                    return None;
                }

                Self::List(MetaItemListParser::new(args, psess, should_emit)?)
            }
            AttrArgs::Eq { eq_span, expr } => Self::NameValue(NameValueParser {
                eq_span: *eq_span,
                value: expr_to_lit(psess, &expr, expr.span, should_emit)?,
                value_span: expr.span,
            }),
        })
    }

    /// Asserts that this MetaItem is a list
    ///
    /// Some examples:
    ///
    /// - `#[allow(clippy::complexity)]`: `(clippy::complexity)` is a list
    /// - `#[rustfmt::skip::macros(target_macro_name)]`: `(target_macro_name)` is a list
    pub fn list(&self) -> Option<&MetaItemListParser<'a>> {
        match self {
            Self::List(l) => Some(l),
            Self::NameValue(_) | Self::NoArgs => None,
        }
    }

    /// Asserts that this MetaItem is a name-value pair.
    ///
    /// Some examples:
    ///
    /// - `#[clippy::cyclomatic_complexity = "100"]`: `clippy::cyclomatic_complexity = "100"` is a name value pair,
    ///   where the name is a path (`clippy::cyclomatic_complexity`). You already checked the path
    ///   to get an `ArgParser`, so this method will effectively only assert that the `= "100"` is
    ///   there
    /// - `#[doc = "hello"]`: `doc = "hello`  is also a name value pair
    pub fn name_value(&self) -> Option<&NameValueParser> {
        match self {
            Self::NameValue(n) => Some(n),
            Self::List(_) | Self::NoArgs => None,
        }
    }

    /// Assert that there were no args.
    /// If there were, get a span to the arguments
    /// (to pass to [`AcceptContext::expected_no_args`](crate::context::AcceptContext::expected_no_args)).
    pub fn no_args(&self) -> Result<(), Span> {
        match self {
            Self::NoArgs => Ok(()),
            Self::List(args) => Err(args.span),
            Self::NameValue(args) => Err(args.eq_span.to(args.value_span)),
        }
    }
}

/// Inside lists, values could be either literals, or more deeply nested meta items.
/// This enum represents that.
///
/// Choose which one you want using the provided methods.
#[derive(Debug, Clone)]
pub enum MetaItemOrLitParser<'a> {
    MetaItemParser(MetaItemParser<'a>),
    Lit(MetaItemLit),
    Err(Span, ErrorGuaranteed),
}

impl<'a> MetaItemOrLitParser<'a> {
    pub fn span(&self) -> Span {
        match self {
            MetaItemOrLitParser::MetaItemParser(generic_meta_item_parser) => {
                generic_meta_item_parser.span()
            }
            MetaItemOrLitParser::Lit(meta_item_lit) => meta_item_lit.span,
            MetaItemOrLitParser::Err(span, _) => *span,
        }
    }

    pub fn lit(&self) -> Option<&MetaItemLit> {
        match self {
            MetaItemOrLitParser::Lit(meta_item_lit) => Some(meta_item_lit),
            _ => None,
        }
    }

    pub fn meta_item(&self) -> Option<&MetaItemParser<'a>> {
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
#[derive(Clone)]
pub struct MetaItemParser<'a> {
    path: PathParser<'a>,
    args: ArgParser<'a>,
}

impl<'a> Debug for MetaItemParser<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaItemParser")
            .field("path", &self.path)
            .field("args", &self.args)
            .finish()
    }
}

impl<'a> MetaItemParser<'a> {
    /// Create a new parser from a [`NormalAttr`], which is stored inside of any
    /// [`ast::Attribute`](rustc_ast::Attribute)
    pub fn from_attr<'sess>(
        attr: &'a NormalAttr,
        parts: &[Symbol],
        psess: &'sess ParseSess,
        should_emit: ShouldEmit,
    ) -> Option<Self> {
        Some(Self {
            path: PathParser(Cow::Borrowed(&attr.item.path)),
            args: ArgParser::from_attr_args(&attr.item.args, parts, psess, should_emit)?,
        })
    }
}

impl<'a> MetaItemParser<'a> {
    pub fn span(&self) -> Span {
        if let Some(other) = self.args.span() {
            self.path.span().with_hi(other.hi())
        } else {
            self.path.span()
        }
    }

    /// Gets just the path, without the args. Some examples:
    ///
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path
    /// - `#[allow(clippy::complexity)]`: `clippy::complexity` is a path
    /// - `#[inline]`: `inline` is a single segment path
    pub fn path(&self) -> &PathParser<'a> {
        &self.path
    }

    /// Gets just the args parser, without caring about the path.
    pub fn args(&self) -> &ArgParser<'a> {
        &self.args
    }

    /// Asserts that this MetaItem starts with a word, or single segment path.
    ///
    /// Some examples:
    /// - `#[inline]`: `inline` is a word
    /// - `#[rustfmt::skip]`: `rustfmt::skip` is a path,
    ///   and not a word and should instead be parsed using [`path`](Self::path)
    pub fn word_is(&self, sym: Symbol) -> Option<&ArgParser<'a>> {
        self.path().word_is(sym).then(|| self.args())
    }
}

#[derive(Clone)]
pub struct NameValueParser {
    pub eq_span: Span,
    value: MetaItemLit,
    pub value_span: Span,
}

impl Debug for NameValueParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NameValueParser")
            .field("eq_span", &self.eq_span)
            .field("value", &self.value)
            .field("value_span", &self.value_span)
            .finish()
    }
}

impl NameValueParser {
    pub fn value_as_lit(&self) -> &MetaItemLit {
        &self.value
    }

    pub fn value_as_str(&self) -> Option<Symbol> {
        self.value_as_lit().kind.str()
    }
}

fn expr_to_lit(
    psess: &ParseSess,
    expr: &Expr,
    span: Span,
    should_emit: ShouldEmit,
) -> Option<MetaItemLit> {
    if let ExprKind::Lit(token_lit) = expr.kind {
        let res = MetaItemLit::from_token_lit(token_lit, expr.span);
        match res {
            Ok(lit) => {
                if token_lit.suffix.is_some() {
                    should_emit.emit_err(
                        psess.dcx().create_err(SuffixedLiteralInAttribute { span: lit.span }),
                    );
                    None
                } else {
                    if !lit.kind.is_unsuffixed() {
                        // Emit error and continue, we can still parse the attribute as if the suffix isn't there
                        should_emit.emit_err(
                            psess.dcx().create_err(SuffixedLiteralInAttribute { span: lit.span }),
                        );
                    }

                    Some(lit)
                }
            }
            Err(err) => {
                let guar = report_lit_error(psess, err, token_lit, expr.span);
                let lit = MetaItemLit {
                    symbol: token_lit.symbol,
                    suffix: token_lit.suffix,
                    kind: LitKind::Err(guar),
                    span: expr.span,
                };
                Some(lit)
            }
        }
    } else {
        if matches!(should_emit, ShouldEmit::Nothing) {
            return None;
        }

        // Example cases:
        // - `#[foo = 1+1]`: results in `ast::ExprKind::BinOp`.
        // - `#[foo = include_str!("nonexistent-file.rs")]`:
        //   results in `ast::ExprKind::Err`. In that case we delay
        //   the error because an earlier error will have already
        //   been reported.
        let msg = "attribute value must be a literal";
        let err = psess.dcx().struct_span_err(span, msg);
        should_emit.emit_err(err);
        None
    }
}

struct MetaItemListParserContext<'a, 'sess> {
    parser: &'a mut Parser<'sess>,
    should_emit: ShouldEmit,
}

impl<'a, 'sess> MetaItemListParserContext<'a, 'sess> {
    fn parse_unsuffixed_meta_item_lit(&mut self) -> PResult<'sess, MetaItemLit> {
        let Some(token_lit) = self.parser.eat_token_lit() else { return Err(self.expected_lit()) };
        self.unsuffixed_meta_item_from_lit(token_lit)
    }

    fn unsuffixed_meta_item_from_lit(
        &mut self,
        token_lit: token::Lit,
    ) -> PResult<'sess, MetaItemLit> {
        let lit = match MetaItemLit::from_token_lit(token_lit, self.parser.prev_token.span) {
            Ok(lit) => lit,
            Err(err) => {
                return Err(create_lit_error(
                    &self.parser.psess,
                    err,
                    token_lit,
                    self.parser.prev_token_uninterpolated_span(),
                ));
            }
        };

        if !lit.kind.is_unsuffixed() {
            // Emit error and continue, we can still parse the attribute as if the suffix isn't there
            self.should_emit.emit_err(
                self.parser.dcx().create_err(SuffixedLiteralInAttribute { span: lit.span }),
            );
        }

        Ok(lit)
    }

    fn parse_attr_item(&mut self) -> PResult<'sess, MetaItemParser<'static>> {
        if let Some(MetaVarKind::Meta { has_meta_form }) = self.parser.token.is_metavar_seq() {
            return if has_meta_form {
                let attr_item = self
                    .parser
                    .eat_metavar_seq(MetaVarKind::Meta { has_meta_form: true }, |this| {
                        MetaItemListParserContext { parser: this, should_emit: self.should_emit }
                            .parse_attr_item()
                    })
                    .unwrap();
                Ok(attr_item)
            } else {
                self.parser.unexpected_any()
            };
        }

        let path = self.parser.parse_path(PathStyle::Mod)?;

        // Check style of arguments that this meta item has
        let args = if self.parser.check(exp!(OpenParen)) {
            let start = self.parser.token.span;
            let (sub_parsers, _) = self.parser.parse_paren_comma_seq(|parser| {
                MetaItemListParserContext { parser, should_emit: self.should_emit }
                    .parse_meta_item_inner()
            })?;
            let end = self.parser.prev_token.span;
            ArgParser::List(MetaItemListParser { sub_parsers, span: start.with_hi(end.hi()) })
        } else if self.parser.eat(exp!(Eq)) {
            let eq_span = self.parser.prev_token.span;
            let value = self.parse_unsuffixed_meta_item_lit()?;

            ArgParser::NameValue(NameValueParser { eq_span, value, value_span: value.span })
        } else {
            ArgParser::NoArgs
        };

        Ok(MetaItemParser { path: PathParser(Cow::Owned(path)), args })
    }

    fn parse_meta_item_inner(&mut self) -> PResult<'sess, MetaItemOrLitParser<'static>> {
        if let Some(token_lit) = self.parser.eat_token_lit() {
            // If a literal token is parsed, we commit to parsing a MetaItemLit for better errors
            Ok(MetaItemOrLitParser::Lit(self.unsuffixed_meta_item_from_lit(token_lit)?))
        } else {
            let prev_pros = self.parser.approx_token_stream_pos();
            match self.parse_attr_item() {
                Ok(item) => Ok(MetaItemOrLitParser::MetaItemParser(item)),
                Err(err) => {
                    // If `parse_attr_item` made any progress, it likely has a more precise error we should prefer
                    // If it didn't make progress we use the `expected_lit` from below
                    if self.parser.approx_token_stream_pos() != prev_pros {
                        Err(err)
                    } else {
                        err.cancel();
                        Err(self.expected_lit())
                    }
                }
            }
        }
    }

    fn expected_lit(&mut self) -> Diag<'sess> {
        let mut err = InvalidMetaItem {
            span: self.parser.token.span,
            descr: token_descr(&self.parser.token),
            quote_ident_sugg: None,
            remove_neg_sugg: None,
        };

        // Suggest quoting idents, e.g. in `#[cfg(key = value)]`. We don't use `Token::ident` and
        // don't `uninterpolate` the token to avoid suggesting anything butchered or questionable
        // when macro metavariables are involved.
        if self.parser.prev_token == token::Eq
            && let token::Ident(..) = self.parser.token.kind
        {
            let before = self.parser.token.span.shrink_to_lo();
            while let token::Ident(..) = self.parser.token.kind {
                self.parser.bump();
            }
            err.quote_ident_sugg = Some(InvalidMetaItemQuoteIdentSugg {
                before,
                after: self.parser.prev_token.span.shrink_to_hi(),
            });
        }

        if self.parser.token == token::Minus
            && self
                .parser
                .look_ahead(1, |t| matches!(t.kind, rustc_ast::token::TokenKind::Literal { .. }))
        {
            err.remove_neg_sugg =
                Some(InvalidMetaItemRemoveNegSugg { negative_sign: self.parser.token.span });
            self.parser.bump();
            self.parser.bump();
        }

        self.parser.dcx().create_err(err)
    }

    fn parse(
        tokens: TokenStream,
        psess: &'sess ParseSess,
        span: Span,
        should_emit: ShouldEmit,
    ) -> PResult<'sess, MetaItemListParser<'static>> {
        let mut parser = Parser::new(psess, tokens, None);
        let mut this = MetaItemListParserContext { parser: &mut parser, should_emit };

        // Presumably, the majority of the time there will only be one attr.
        let mut sub_parsers = ThinVec::with_capacity(1);
        while this.parser.token != token::Eof {
            sub_parsers.push(this.parse_meta_item_inner()?);

            if !this.parser.eat(exp!(Comma)) {
                break;
            }
        }

        if parser.token != token::Eof {
            parser.unexpected()?;
        }

        Ok(MetaItemListParser { sub_parsers, span })
    }
}

#[derive(Debug, Clone)]
pub struct MetaItemListParser<'a> {
    sub_parsers: ThinVec<MetaItemOrLitParser<'a>>,
    pub span: Span,
}

impl<'a> MetaItemListParser<'a> {
    fn new<'sess>(
        delim: &'a DelimArgs,
        psess: &'sess ParseSess,
        should_emit: ShouldEmit,
    ) -> Option<Self> {
        match MetaItemListParserContext::parse(
            delim.tokens.clone(),
            psess,
            delim.dspan.entire(),
            should_emit,
        ) {
            Ok(s) => Some(s),
            Err(e) => {
                should_emit.emit_err(e);
                None
            }
        }
    }

    /// Lets you pick and choose as what you want to parse each element in the list
    pub fn mixed(&self) -> impl Iterator<Item = &MetaItemOrLitParser<'a>> {
        self.sub_parsers.iter()
    }

    pub fn len(&self) -> usize {
        self.sub_parsers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns Some if the list contains only a single element.
    ///
    /// Inside the Some is the parser to parse this single element.
    pub fn single(&self) -> Option<&MetaItemOrLitParser<'a>> {
        let mut iter = self.mixed();
        iter.next().filter(|_| iter.next().is_none())
    }
}
