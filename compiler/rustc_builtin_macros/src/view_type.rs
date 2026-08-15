use rustc_ast::token::TokenKind;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{Ty, ast};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_parse::parser::{ExpTokenPair, TokenType};
use rustc_span::{Ident, Span};
use thin_vec::ThinVec;

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let (ty, pat) = match parse_view_ty(cx, tts) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ExpandResult::Ready(DummyResult::any(sp, err.emit()));
        }
    };

    ExpandResult::Ready(base::MacEager::ty(cx.ty(sp, ast::TyKind::View(ty, pat))))
}

fn parse_view_ty<'a>(
    cx: &mut ExtCtxt<'a>,
    stream: TokenStream,
) -> PResult<'a, (Box<Ty>, ThinVec<Ident>)> {
    let mut parser = cx.new_parser_from_tts(stream);

    let ty = parser.parse_ty()?;

    parser.expect(ExpTokenPair { tok: TokenKind::Dot, token_type: TokenType::Dot })?;

    let fields = match parser.parse_delim_comma_seq(
        ExpTokenPair { tok: TokenKind::OpenBrace, token_type: TokenType::OpenBrace },
        ExpTokenPair { tok: TokenKind::CloseBrace, token_type: TokenType::CloseBrace },
        |p| p.parse_field_name(),
    ) {
        Ok((fields, _)) => fields,
        Err(diag) => {
            return Err(diag);
        }
    };

    Ok((ty, fields))
}
