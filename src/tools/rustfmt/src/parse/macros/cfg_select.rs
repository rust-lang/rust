//! See [`cfg_select!` reference](
//! https://doc.rust-lang.org/nightly/reference/conditional-compilation.html#the-cfg_select-macro
//! ) for grammar.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rustc_ast::ast;
use rustc_ast::token;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_parse::exp;
use rustc_parse::parser::{AllowConstBlockItems, ForceCollect};
use rustc_span::Span;
use tracing::debug;

use crate::comment::contains_comment;
use crate::expr::is_unsafe_block;
use crate::parse::macros::build_stream_parser;
use crate::parse::session::ParseSess;
use crate::rewrite::RewriteContext;
use crate::spanned::Spanned;

// NB: parser recovery can return `Ok(expr)` (or other non-terminals) and still emit errors. We need
// to properly reset the shared error count.

pub(crate) fn parse_items_from_cfg_select<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        parse_items_from_cfg_select_inner(psess, mac)
    }));
    let dcx = psess.inner().dcx();
    let recovered = dcx.has_errors().is_some();
    dcx.reset_err_count();
    match result {
        Ok(Ok(_)) if recovered => Err("cfg_select! parsed with recovery"),
        Ok(res) => res,
        Err(..) => Err("failed to parse cfg_select!"),
    }
}

fn parse_items_from_cfg_select_inner<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    let ts = mac.args.tokens.clone();
    let mut parser = build_stream_parser(psess.inner(), ts);

    if parser.token == TokenKind::OpenBrace {
        return Err("Expression position cfg_select! not yet supported");
    }

    let mut items = vec![];

    while parser.token.kind != TokenKind::Eof {
        if !parser.eat_keyword(exp!(Underscore)) {
            parser.parse_attr_item(ForceCollect::No).map_err(|e| {
                e.cancel();
                "Failed to parse attr item"
            })?;
        }

        if !parser.eat(exp!(FatArrow)) {
            return Err("Expected a fat arrow");
        }

        if !parser.eat(exp!(OpenBrace)) {
            return Err("Expected an opening brace");
        }

        while parser.token != TokenKind::CloseBrace && parser.token.kind != TokenKind::Eof {
            let item = match parser
                .parse_item(ForceCollect::No, AllowConstBlockItems::DoesNotMatter)
            {
                Ok(Some(item_ptr)) => *item_ptr,
                Ok(None) => {
                    // Advance the parser by at least one token to prevent an infinite loop
                    parser.bump();
                    continue;
                }
                Err(err) => {
                    err.cancel();
                    return Err(
                        "Expected item inside cfg_select block, but failed to parse it as an item",
                    );
                }
            };
            if let ast::ItemKind::Mod(..) = item.kind {
                items.push(item);
            }
        }

        if !parser.eat(exp!(CloseBrace)) {
            return Err("Expected a closing brace");
        }

        if parser.eat(exp!(Eof)) {
            break;
        }
    }

    Ok(items)
}

/// LHS predicate of a `cfg_select!` arm.
pub(crate) enum CfgSelectFormatPredicate {
    /// Example: the `unix` in `unix => {}`. Notably, outer or inner attributes are not permitted.
    Cfg(ast::MetaItemInner),
    /// `_` in `_ => {}`.
    Wildcard(Span),
}

impl Spanned for CfgSelectFormatPredicate {
    fn span(&self) -> rustc_span::Span {
        match self {
            Self::Cfg(meta_item_inner) => meta_item_inner.span(),
            Self::Wildcard(span) => *span,
        }
    }
}

/// Each `$predicate => $production` arm in `cfg_select!`.
pub(crate) struct CfgSelectArm {
    /// The `$predicate` part.
    pub(crate) predicate: CfgSelectFormatPredicate,
    /// Span of `=>`.
    pub(crate) arrow: Token,
    /// The RHS `$production` expression.
    pub(crate) expr: Box<ast::Expr>,
    /// `cfg_select!` arms `$production`s can be optionally `,` terminated, like `match` arms.
    /// The `,` is not needed when `$production` is itself braced `{}`.
    pub(crate) trailing_comma: Option<Span>,
}

impl PartialEq for &CfgSelectArm {
    fn eq(&self, other: &Self) -> bool {
        // consider the arms equal if they have the same span
        self.span() == other.span()
    }
}

impl Spanned for CfgSelectArm {
    fn span(&self) -> Span {
        self.predicate
            .span()
            .with_hi(if let Some(comma) = self.trailing_comma {
                comma.hi()
            } else {
                self.expr.span.hi()
            })
    }
}

impl std::fmt::Debug for CfgSelectArm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.predicate {
            CfgSelectFormatPredicate::Cfg(cfg_entry) => cfg_entry.fmt(f)?,
            CfgSelectFormatPredicate::Wildcard(t) => t.fmt(f)?,
        };
        write!(f, "=> {:?}", self.expr)
    }
}

impl CfgSelectArm {
    /// Is the cfg_select! arm written with two consecutive blocks.
    /// For example `$predicate => {{}}`. If so, extract the inner block.
    /// rustfmt will preserve the double block syntax.
    pub(crate) fn double_block_inner(&self, context: &RewriteContext<'_>) -> Option<&ast::Expr> {
        // In order to see if we can reach in and grab the inner block we'll
        // check the following properties of the outer block:
        // 1. It has a single statement that's also a block
        // 2. It doesn't have a label
        // 3. It isn't an unsafe block
        // 4. It doesn't have any attributes
        match &self.expr.kind {
            ast::ExprKind::Block(b, label)
                if b.stmts.len() == 1
                    && label.is_none()
                    && !is_unsafe_block(b)
                    && self.expr.attrs.is_empty() =>
            {
                match &b.stmts[0].kind {
                    // This is where we check that the inner block's statement is also a block
                    ast::StmtKind::Expr(e) if matches!(e.kind, ast::ExprKind::Block(..)) => {
                        let pre_span = b.span.shrink_to_lo().between(e.span());
                        let pre_snippet = context.snippet(pre_span).trim();

                        let post_span = e.span().shrink_to_hi().between(b.span.shrink_to_hi());
                        let post_snippet = context.snippet(post_span).trim();

                        if !contains_comment(pre_snippet) && !contains_comment(post_snippet) {
                            // To prevent comment loss between the blocks we only want to return
                            // the inner block if there aren't any pre / post comments.
                            Some(e)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// FIXME(ytmimi) would be nice if rustfmt didn't need to implement parsing logic on its own
// and could instead just call rustc_attr_parsing::parse_cfg_select, but this is fine for now.
pub(crate) fn parse_cfg_select_arms(
    psess: &ParseSess,
    ts: TokenStream,
) -> Option<Vec<CfgSelectArm>> {
    let result = catch_unwind(AssertUnwindSafe(|| parse_cfg_select_arms_inner(psess, ts)));
    let dcx = psess.inner().dcx();
    let recovered = dcx.has_errors().is_some();
    dcx.reset_err_count();
    match result {
        Ok(Some(_)) if recovered => {
            debug!("cfg_select! arms parsed with recovery");
            None
        }
        Ok(arms) => arms,
        Err(..) => {
            debug!("failed to parse cfg_select! arms");
            None
        }
    }
}

fn parse_cfg_select_arms_inner(psess: &ParseSess, ts: TokenStream) -> Option<Vec<CfgSelectArm>> {
    let mut cfg_select_predicates = vec![];
    let mut parser = build_stream_parser(psess.inner(), ts);

    while parser.token != token::Eof {
        let predicate = if parser.eat_keyword(exp!(Underscore)) {
            CfgSelectFormatPredicate::Wildcard(parser.prev_token.span)
        } else {
            let Ok(meta_item) = parser.parse_meta_item_inner().map_err(|e| e.cancel()) else {
                debug!("Failed to parse cfg entry in cfg_select! predicate");
                return None;
            };

            CfgSelectFormatPredicate::Cfg(meta_item)
        };

        if let Err(e) = parser.expect(exp!(FatArrow)) {
            e.cancel();
            debug!("Expected to find a `=>` after cfg_select! predicate.");
            return None;
        };

        let arrow = parser.prev_token;

        let Ok(expr) = parser.parse_expr().map_err(|e| e.cancel()) else {
            debug!("Couldn't parse cfg_select! arm body after `=>`.");
            return None;
        };

        let trailing_comma = if parser.eat(exp!(Comma)) {
            Some(parser.prev_token.span)
        } else {
            None
        };

        let arm = CfgSelectArm {
            predicate,
            arrow,
            expr,
            trailing_comma,
        };

        cfg_select_predicates.push(arm);
    }
    Some(cfg_select_predicates)
}
