use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{Pat, Ty, ast};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::{Span, sym};

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let (ty, pat) = match parse_pat_ty(cx, tts) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ExpandResult::Ready(DummyResult::any(sp, err.emit()));
        }
    };

    ExpandResult::Ready(base::MacEager::ty(cx.ty(sp, ast::TyKind::Pat(ty, pat))))
}

fn parse_pat_ty<'a>(cx: &mut ExtCtxt<'a>, stream: TokenStream) -> PResult<'a, (P<Ty>, P<Pat>)> {
    let mut parser = cx.new_parser_from_tts(stream);

    let ty = parser.parse_ty()?;
    parser.expect_keyword(sym::is)?;
    let pat = parser.parse_pat_no_top_alt(None, None)?;

    Ok((ty, pat))
}
