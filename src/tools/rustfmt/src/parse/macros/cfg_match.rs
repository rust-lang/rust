use std::panic::{AssertUnwindSafe, catch_unwind};

use rustc_ast::ast;
use rustc_ast::token::TokenKind;
use rustc_parse::exp;
use rustc_parse::parser::{AllowConstBlockItems, ForceCollect};

use crate::parse::macros::build_stream_parser;
use crate::parse::session::ParseSess;

pub(crate) fn parse_cfg_match<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    match catch_unwind(AssertUnwindSafe(|| parse_cfg_match_inner(psess, mac))) {
        Ok(Ok(items)) => Ok(items),
        Ok(err @ Err(_)) => err,
        Err(..) => Err("failed to parse cfg_match!"),
    }
}

fn parse_cfg_match_inner<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    let ts = mac.args.tokens.clone();
    let mut parser = build_stream_parser(psess.inner(), ts);

    if parser.token == TokenKind::OpenBrace {
        return Err("Expression position cfg_match! not yet supported");
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
                Ok(None) => continue,
                Err(err) => {
                    err.cancel();
                    parser.psess.dcx().reset_err_count();
                    return Err(
                        "Expected item inside cfg_match block, but failed to parse it as an item",
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
