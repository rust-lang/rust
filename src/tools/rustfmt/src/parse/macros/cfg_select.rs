use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    ast,
    token::{self, Token},
};
use rustc_parse::exp;
use rustc_span::Span;
use tracing::debug;

use crate::parse::session::ParseSess;
use crate::spanned::Spanned;

pub(crate) enum CfgSelectFormatPredicate {
    Cfg(ast::MetaItemInner),
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

pub(crate) struct CfgSelectArm {
    pub(crate) predicate: CfgSelectFormatPredicate,
    pub(crate) arrow: Token,
    pub(crate) expr: Box<ast::Expr>,
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

// FIXME(ytmimi) would be nice if rustfmt didn't need to implement parsing logic on its own
// and could instead just call rustc_attr_parsing::parse_cfg_select, but this is fine for now.
pub(crate) fn parse_cfg_select(psess: &ParseSess, ts: TokenStream) -> Option<Vec<CfgSelectArm>> {
    let mut cfg_select_predicates = vec![];
    let mut parser = super::build_stream_parser(psess.inner(), ts);

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

        if let Err(_) = parser.expect(exp!(FatArrow)) {
            debug!("Expected to find a `=>` after cfg_selec! predicate.");
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
