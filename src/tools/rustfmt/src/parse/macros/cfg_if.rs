use std::panic::{AssertUnwindSafe, catch_unwind};

use rustc_ast::ast;
use rustc_ast::token::TokenKind;
use rustc_parse::exp;
use rustc_parse::parser::ForceCollect;
use rustc_span::symbol::kw;

use crate::parse::macros::build_stream_parser;
use crate::parse::session::ParseSess;

pub(crate) fn parse_cfg_if<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    match catch_unwind(AssertUnwindSafe(|| parse_cfg_if_inner(psess, mac))) {
        Ok(Ok(items)) => Ok(items),
        Ok(err @ Err(_)) => err,
        Err(..) => Err("failed to parse cfg_if!"),
    }
}

fn parse_cfg_if_inner<'a>(
    psess: &'a ParseSess,
    mac: &'a ast::MacCall,
) -> Result<Vec<ast::Item>, &'static str> {
    let ts = mac.args.tokens.clone();
    let mut parser = build_stream_parser(psess.inner(), ts);

    let mut items = vec![];
    let mut process_if_cfg = true;

    while parser.token.kind != TokenKind::Eof {
        if process_if_cfg {
            if !parser.eat_keyword(exp!(If)) {
                return Err("Expected `if`");
            }

            if !matches!(parser.token.kind, TokenKind::Pound) {
                return Err("Failed to parse attributes");
            }

            // Inner attributes are not actually syntactically permitted here, but we don't
            // care about inner vs outer attributes in this position. Our purpose with this
            // special case parsing of cfg_if macros is to ensure we can correctly resolve
            // imported modules that may have a custom `path` defined.
            //
            // As such, we just need to advance the parser past the attribute and up to
            // to the opening brace.
            // See also https://github.com/rust-lang/rust/pull/79433
            parser
                .parse_attribute(rustc_parse::parser::attr::InnerAttrPolicy::Permitted)
                .map_err(|e| {
                    e.cancel();
                    "Failed to parse attributes"
                })?;
        }

        if !parser.eat(exp!(OpenBrace)) {
            return Err("Expected an opening brace");
        }

        while parser.token != TokenKind::CloseBrace && parser.token.kind != TokenKind::Eof {
            let item = match parser.parse_item(ForceCollect::No) {
                Ok(Some(item_ptr)) => *item_ptr,
                Ok(None) => continue,
                Err(err) => {
                    err.cancel();
                    parser.psess.dcx().reset_err_count();
                    return Err(
                        "Expected item inside cfg_if block, but failed to parse it as an item",
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

        if !parser.eat_keyword(exp!(Else)) {
            return Err("Expected `else`");
        }

        process_if_cfg = parser.token.is_keyword(kw::If);
    }

    Ok(items)
}
