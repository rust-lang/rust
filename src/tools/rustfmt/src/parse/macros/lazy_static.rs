use rustc_ast::ast;
use rustc_ast::ptr::P;
use rustc_ast::token::TokenKind;
use rustc_ast::tokenstream::TokenStream;
use rustc_span::symbol::{self, kw};

use crate::rewrite::RewriteContext;

pub(crate) fn parse_lazy_static(
    context: &RewriteContext<'_>,
    ts: TokenStream,
) -> Option<Vec<(ast::Visibility, symbol::Ident, P<ast::Ty>, P<ast::Expr>)>> {
    let mut result = vec![];
    let mut parser = super::build_parser(context, ts);
    macro_rules! parse_or {
        ($method:ident $(,)* $($arg:expr),* $(,)*) => {
            match parser.$method($($arg,)*) {
                Ok(val) => {
                    if parser.psess.dcx().has_errors().is_some() {
                        parser.psess.dcx().reset_err_count();
                        return None;
                    } else {
                        val
                    }
                }
                Err(err) => {
                    err.cancel();
                    parser.psess.dcx().reset_err_count();
                    return None;
                }
            }
        }
    }
    while parser.token.kind != TokenKind::Eof {
        // Parse a `lazy_static!` item.
        // FIXME: These `eat_*` calls should be converted to `parse_or` to avoid
        // silently formatting malformed lazy-statics.
        let vis = parse_or!(parse_visibility, rustc_parse::parser::FollowedByType::No);
        let _ = parser.eat_keyword(kw::Static);
        let _ = parser.eat_keyword(kw::Ref);
        let id = parse_or!(parse_ident);
        let _ = parser.eat(&TokenKind::Colon);
        let ty = parse_or!(parse_ty);
        let _ = parser.eat(&TokenKind::Eq);
        let expr = parse_or!(parse_expr);
        let _ = parser.eat(&TokenKind::Semi);
        result.push((vis, id, ty, expr));
    }

    Some(result)
}
