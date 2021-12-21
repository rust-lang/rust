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
                    if parser.sess.span_diagnostic.has_errors() {
                        parser.sess.span_diagnostic.reset_err_count();
                        return None;
                    } else {
                        val
                    }
                }
                Err(mut err) => {
                    err.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                    return None;
                }
            }
        }
    }

    while parser.token.kind != TokenKind::Eof {
        // Parse a `lazy_static!` item.
        let vis = parse_or!(parse_visibility, rustc_parse::parser::FollowedByType::No);
        parser.eat_keyword(kw::Static);
        parser.eat_keyword(kw::Ref);
        let id = parse_or!(parse_ident);
        parser.eat(&TokenKind::Colon);
        let ty = parse_or!(parse_ty);
        parser.eat(&TokenKind::Eq);
        let expr = parse_or!(parse_expr);
        parser.eat(&TokenKind::Semi);
        result.push((vis, id, ty, expr));
    }

    Some(result)
}
