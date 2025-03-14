use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AnonConst, DUMMY_NODE_ID, Ty, TyPat, TyPatKind, ast};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_parse::exp;
use rustc_span::Span;

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

fn parse_pat_ty<'a>(cx: &mut ExtCtxt<'a>, stream: TokenStream) -> PResult<'a, (P<Ty>, P<TyPat>)> {
    let mut parser = cx.new_parser_from_tts(stream);

    let ty = parser.parse_ty()?;
    parser.expect_keyword(exp!(Is))?;
    let pat = parser.parse_pat_no_top_alt(None, None)?.into_inner();

    let kind = match pat.kind {
        ast::PatKind::Range(start, end, include_end) => TyPatKind::Range(
            start.map(|value| P(AnonConst { id: DUMMY_NODE_ID, value })),
            end.map(|value| P(AnonConst { id: DUMMY_NODE_ID, value })),
            include_end,
        ),
        ast::PatKind::Err(guar) => TyPatKind::Err(guar),
        _ => TyPatKind::Err(cx.dcx().span_err(pat.span, "pattern not supported in pattern types")),
    };

    let pat = P(TyPat { id: pat.id, kind, span: pat.span, tokens: pat.tokens });

    Ok((ty, pat))
}
