use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{CoroutineKind, CoroutineMarker, Expr, ast, token};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::Span;

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let closure = match parse_closure(cx, sp, tts) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ExpandResult::Ready(DummyResult::any(sp, err.emit()));
        }
    };

    ExpandResult::Ready(base::MacEager::expr(closure))
}

fn parse_closure<'a>(
    cx: &mut ExtCtxt<'a>,
    span: Span,
    stream: TokenStream,
) -> PResult<'a, Box<Expr>> {
    let mut closure_parser = cx.new_parser_from_tts(stream);

    let coroutine_marker = Some(CoroutineMarker::new(CoroutineKind::Gen, span));

    let mut closure = closure_parser.parse_expr()?;
    match &mut closure.kind {
        ast::ExprKind::Closure(c) => {
            if let Some(marker) = c.coroutine_marker {
                cx.dcx().span_err(marker.span, "only plain closures allowed in `iter!`");
            }
            c.coroutine_marker = coroutine_marker;
            if closure_parser.token != token::Eof {
                closure_parser.unexpected()?;
            }
            Ok(closure)
        }
        _ => {
            cx.dcx().span_err(closure.span, "`iter!` body must be a closure");
            Err(closure_parser.unexpected().unwrap_err())
        }
    }
}
