use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AnonConst, DUMMY_NODE_ID, Ty, TyPat, TyPatKind, ast, token};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_parse::exp;
use rustc_parse::parser::{CommaRecoveryMode, RecoverColon, RecoverComma};
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

fn parse_pat_ty<'a>(
    cx: &mut ExtCtxt<'a>,
    stream: TokenStream,
) -> PResult<'a, (Box<Ty>, Box<TyPat>)> {
    let mut parser = cx.new_parser_from_tts(stream);

    let ty = parser.parse_ty()?;
    parser.expect_keyword(exp!(Is))?;

    let pat = pat_to_ty_pat(
        cx,
        *parser.parse_pat_no_top_guard(
            None,
            RecoverComma::No,
            RecoverColon::No,
            CommaRecoveryMode::EitherTupleOrPipe,
        )?,
    );

    if parser.token != token::Eof {
        parser.unexpected()?;
    }

    Ok((ty, pat))
}

fn ty_pat(kind: TyPatKind, span: Span) -> Box<TyPat> {
    Box::new(TyPat { id: DUMMY_NODE_ID, kind, span, tokens: None })
}

fn pat_to_ty_pat(cx: &mut ExtCtxt<'_>, pat: ast::Pat) -> Box<TyPat> {
    let kind = match pat.kind {
        ast::PatKind::Range(start, end, include_end) => TyPatKind::Range(
            start.map(|value| Box::new(AnonConst { id: DUMMY_NODE_ID, value })),
            end.map(|value| Box::new(AnonConst { id: DUMMY_NODE_ID, value })),
            include_end,
        ),
        ast::PatKind::Or(variants) => {
            TyPatKind::Or(variants.into_iter().map(|pat| pat_to_ty_pat(cx, *pat)).collect())
        }
        ast::PatKind::Err(guar) => TyPatKind::Err(guar),
        _ => TyPatKind::Err(cx.dcx().span_err(pat.span, "pattern not supported in pattern types")),
    };
    ty_pat(kind, pat.span)
}
